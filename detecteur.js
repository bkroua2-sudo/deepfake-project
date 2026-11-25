
const MODEL_URL =
  "https://huggingface.co/TensorFold/detectra-v1/resolve/main/model.onnx";

const ORT_URL =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js";

let detectorSession = null;

async function loadRuntime() {
  if (window.ort) return;

  await new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = ORT_URL;
    script.onload = resolve;
    script.onerror = () =>
      reject(new Error("Impossible de charger ONNX Runtime."));
    document.head.appendChild(script);
  });

  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
}

async function loadDetector() {
  if (detectorSession) return detectorSession;

  const preview = document.getElementById("inspectPreview");
  preview.textContent = "Chargement du modèle... Patiente un instant.";

  await loadRuntime();

  detectorSession = await ort.InferenceSession.create(
    MODEL_URL,
    {
      executionProviders: ["webgpu", "wasm"]
    }
  );

  return detectorSession;
}

function prepareImage(bitmap) {
  const scale = 440 / Math.min(bitmap.width, bitmap.height);

  const resizedWidth = Math.round(bitmap.width * scale);
  const resizedHeight = Math.round(bitmap.height * scale);

  const resizedCanvas = document.createElement("canvas");
  resizedCanvas.width = resizedWidth;
  resizedCanvas.height = resizedHeight;

  const resizedContext = resizedCanvas.getContext("2d");
  resizedContext.imageSmoothingEnabled = true;
  resizedContext.imageSmoothingQuality = "high";
  resizedContext.drawImage(bitmap, 0, 0, resizedWidth, resizedHeight);

  const left = Math.floor((resizedWidth - 384) / 2);
  const top = Math.floor((resizedHeight - 384) / 2);

  const pixels = resizedContext.getImageData(left, top, 384, 384).data;

  const input = new Float32Array(3 * 384 * 384);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < 384 * 384; i++) {
    for (let c = 0; c < 3; c++) {
      const value = pixels[i * 4 + c] / 255;
      input[c * 384 * 384 + i] =
        (value - mean[c]) / std[c];
    }
  }

  return input;
}

async function analyzeImage(file) {
  const preview = document.getElementById("inspectPreview");

  try {
    preview.textContent = "Préparation de l'image...";

    const bitmap = await createImageBitmap(file);
    const inputData = prepareImage(bitmap);
    bitmap.close();

    const session = await loadDetector();

    preview.textContent = "Analyse en cours...";

    const tensor = new ort.Tensor(
      "float32",
      inputData,
      [1, 3, 384, 384]
    );

    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const results = await session.run({
      [inputName]: tensor
    });

    const logit = results[outputName].data[0];
    const probability = 1 / (1 + Math.exp(-logit));
    const percentage = Math.round(probability * 100);

    const verdict =
      probability >= 0.5
        ? "Cette image semble générée par l'IA."
        : "Cette image semble authentique.";

    preview.textContent =
      verdict +
      " Score estimé : " +
      percentage +
      "% de probabilité IA." +
      " Attention : ce résultat n'est pas une preuve certaine.";

  } catch (error) {
    console.error(error);
    preview.textContent =
      "Une erreur est survenue : " + error.message;
  }
}

const inspectFile = document.getElementById("inspectFile");

if (inspectFile) {
  inspectFile.addEventListener("change", (event) => {
    const file = event.target.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      document.getElementById("inspectPreview").textContent =
        "Choisis un fichier image.";
      return;
    }

    analyzeImage(file);
  });
}
