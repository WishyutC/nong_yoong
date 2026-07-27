from __future__ import annotations

import importlib.util
import unittest

from thai_voice_robot.client.activation import TFLiteWakeWordActivator
from thai_voice_robot.config import PROJECT_ROOT


MODEL_PATH = PROJECT_ROOT / "models" / "keyword_model.tflite"
HAS_RUNTIME = importlib.util.find_spec("tflite_runtime") is not None


@unittest.skipUnless(MODEL_PATH.is_file(), "wake-word model not copied")
@unittest.skipUnless(HAS_RUNTIME, "TFLite runtime not installed")
class WakeModelIntegrationTests(unittest.TestCase):
    def test_model_loads_and_runs_expected_mfcc_tensor(self) -> None:
        import numpy as np

        interpreter = TFLiteWakeWordActivator(MODEL_PATH)._load()
        input_detail = interpreter.get_input_details()[0]
        self.assertEqual(tuple(input_detail["shape"]), (1, 13, 32, 1))
        tensor = np.zeros(input_detail["shape"], dtype=input_detail["dtype"])
        interpreter.set_tensor(input_detail["index"], tensor)
        interpreter.invoke()
        probability = float(
            np.asarray(
                interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
            ).reshape(-1)[0]
        )
        self.assertTrue(np.isfinite(probability))


if __name__ == "__main__":
    unittest.main()
