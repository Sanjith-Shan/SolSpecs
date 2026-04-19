"""
SolSpecs — EMG Bridge
Runs the Mindrove HD classifier in a background thread and fires callbacks
when gestures are confirmed.
"""

import threading


class EMGBridge:
    """
    Real Mindrove bridge: loads calibration, trains HD model, connects to LSL,
    classifies live EMG. Fires on_clench / on_half_clench callbacks.
    """

    def __init__(self):
        self.on_clench = None       # callback: () -> None
        self.on_half_clench = None  # callback: () -> None
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="EMGBridge")
        self._thread.start()

    def _run(self):
        try:
            from core.classify import (
                load_calibration, HDModel, HD_DIM, N_CLASSES, CHANNELS,
                extract_features, window_rms,
                ACTIVITY_THRESHOLD, SPIKE_RMS_MULTIPLIER,
                WINDOW_SIZE, WINDOW_STEP, GESTURE_LABELS, CONFIRM_WINDOWS,
            )
            from pylsl import StreamInlet, resolve_streams
            from collections import deque
            import numpy as np
            import torch

            print("[EMG] Loading calibration data...")
            X, y = load_calibration()
            n_features = X.shape[1]
            model = HDModel(n_features=n_features, n_classes=N_CLASSES, dim=HD_DIM)
            model.train(torch.tensor(X), torch.tensor(y))
            baseline_rms = float(np.sqrt(np.mean(X[:, :CHANNELS] ** 2)))
            spike_threshold = baseline_rms * SPIKE_RMS_MULTIPLIER
            print(f"[EMG] Model trained. Baseline RMS={baseline_rms:.4f}, spike gate={spike_threshold:.4f}")

            streams = resolve_streams()
            if not streams:
                print("[EMG] No LSL stream found. EMG disabled.")
                return
            inlet = StreamInlet(streams[0])
            print("[EMG] Connected to Mindrove LSL stream.")

            buffer = deque(maxlen=WINDOW_SIZE)
            step_counter = 0
            label_history = deque(maxlen=CONFIRM_WINDOWS)
            last_fired = ""

            while self._running:
                sample, _ = inlet.pull_sample(timeout=0.1)
                if not sample:
                    continue
                buffer.append(sample)
                step_counter += 1

                if len(buffer) < WINDOW_SIZE or step_counter % WINDOW_STEP != 0:
                    continue

                window = np.array(buffer, dtype=np.float32)
                rms = window_rms(window)

                if rms < ACTIVITY_THRESHOLD:
                    label_history.clear()
                    last_fired = ""
                    continue

                if rms > spike_threshold:
                    label = "General"
                else:
                    feat = extract_features(window)
                    idx = model.predict(torch.tensor(feat))
                    label = GESTURE_LABELS[idx]

                label_history.append(label)
                all_same = len(set(label_history)) == 1
                buffer_full = len(label_history) == CONFIRM_WINDOWS
                new_label = label != last_fired

                if buffer_full and all_same and new_label:
                    print(f"[EMG] Gesture confirmed: {label.upper()}")
                    last_fired = label
                    if label == "Clench" and self.on_clench:
                        self.on_clench()
                    elif label == "Half_Clench" and self.on_half_clench:
                        self.on_half_clench()
                elif not all_same:
                    last_fired = ""

        except ImportError as e:
            print(f"[EMG] Missing dependency: {e}. EMG disabled.")
        except Exception as e:
            print(f"[EMG] Error: {e}. EMG disabled.")

    def stop(self):
        self._running = False


class MockEMGBridge:
    """
    Keyboard-driven mock for testing without Mindrove hardware.
    Type 'c' + Enter → Clench (fuel scan)
    Type 'h' + Enter → Half-Clench (MAYDAY)
    """

    def __init__(self):
        self.on_clench = None
        self.on_half_clench = None
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="MockEMGBridge")
        self._thread.start()
        print("[EMG-Mock] Press C + Enter for Clench (fuel scan), H + Enter for Half-Clench (MAYDAY)")

    def _run(self):
        import sys
        import select
        while self._running:
            try:
                ready = select.select([sys.stdin], [], [], 0.1)[0]
                if not ready:
                    continue
                key = sys.stdin.readline().strip().lower()
                if key == 'c' and self.on_clench:
                    print("[EMG-Mock] Clench → Fuel Scan")
                    self.on_clench()
                elif key == 'h' and self.on_half_clench:
                    print("[EMG-Mock] Half-Clench → MAYDAY")
                    self.on_half_clench()
            except Exception:
                pass

    def stop(self):
        self._running = False
