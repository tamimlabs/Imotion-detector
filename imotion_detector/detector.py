"""Face detection + emotion analysis + annotation pipeline."""

import logging

import cv2

logger = logging.getLogger(__name__)

HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceEmotionPipeline:
    """Detects faces, analyses their emotion with a pluggable engine and
    annotates the frame with bounding boxes and emotion labels.
    """

    def __init__(
        self,
        engine,
        frame_skip=10,
        min_face_size=100,
        scale_factor=1.1,
        min_neighbors=5,
        pad_ratio=0.2,
    ):
        self.engine = engine
        self.frame_skip = max(1, frame_skip)
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.pad_ratio = pad_ratio

        self._cascade = cv2.CascadeClassifier(HAAR_CASCADE)
        if self._cascade.empty():
            raise RuntimeError("Failed to load the Haar cascade for face detection.")

        self._last_emotions = {}
        self._frame_count = 0

    # ------------------------------------------------------------------ #
    # Face detection                                                      #
    # ------------------------------------------------------------------ #
    def detect_faces(self, gray):
        """Return face boxes as an (N, 4) array of (x, y, w, h)."""
        return self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
        )

    # ------------------------------------------------------------------ #
    # Emotion analysis                                                    #
    # ------------------------------------------------------------------ #
    def _crop_face(self, frame, box):
        """Extract and lightly pad a face crop for the emotion engine."""
        x, y, w, h = (int(v) for v in box)
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        crop = frame[y : y + h, x : x + w]
        if self.pad_ratio > 0 and crop.size:
            px, py = int(w * self.pad_ratio), int(h * self.pad_ratio)
            crop = cv2.copyMakeBorder(crop, py, py, px, px, cv2.BORDER_REPLICATE)
        return crop

    def analyze_frame(self, frame, faces):
        """Run the engine on every face box and return {box: emotion}."""
        results = {}
        for box in faces:
            try:
                emotion = self.engine.analyze(self._crop_face(frame, box))
                results[tuple(box)] = emotion if emotion else "Unknown"
            except Exception as exc:
                logger.debug("Emotion analysis failed for face %s: %s", box, exc)
                results[tuple(box)] = "Processing..."
        return results

    # ------------------------------------------------------------------ #
    # Annotation                                                          #
    # ------------------------------------------------------------------ #
    def _match_emotion(self, x, y, w, h):
        """Find the closest previously analysed face and reuse its label."""
        best, best_dist = None, float("inf")
        for (kx, ky, kw, kh), emotion in self._last_emotions.items():
            dist = abs(x - kx) + abs(y - ky)
            if dist < best_dist:
                best, best_dist = emotion, dist
        if best is not None and best_dist < 50:
            return best
        return "Analyzing..."

    def annotate(self, frame, faces):
        for x, y, w, h in faces:
            emotion = self._match_emotion(x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )

    # ------------------------------------------------------------------ #
    # Per-frame processing                                                #
    # ------------------------------------------------------------------ #
    def process(self, frame):
        """Detect faces, refresh emotions every ``frame_skip`` frames and
        annotate the frame in place. Returns the annotated frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)

        if self._frame_count % self.frame_skip == 0:
            new_emotions = self.analyze_frame(frame, faces)
            if new_emotions:
                self._last_emotions = new_emotions

        self._frame_count += 1
        self.annotate(frame, faces)
        return frame

    def reset(self):
        """Forget cached emotions and frame counter (e.g. before a new input)."""
        self._last_emotions = {}
        self._frame_count = 0
