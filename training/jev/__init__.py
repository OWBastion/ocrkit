"""Offline Jev-Omni dataset/evaluation sidecar experiment (#25).

Replays Studio-reviewed ROI rows through a bounded typed-choice decision task
to measure whether Jev can safely reduce human pre-review: a high-confidence
candidate auto-accepts with that transcription, a high-confidence not-valid
auto-rejects, and everything else routes to human review. The experiment is
read-only against Studio batches, keeps its model dependency removable, and
never feeds Jev output back into labels, training, or production recognition.
"""
