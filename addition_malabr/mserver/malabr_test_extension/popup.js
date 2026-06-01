import LogisticRegression from "./malabr_js/logistic_regression.js";
// import RandomForestClassifier from "./malabr_js/random_forest_classifier.js";
// import SVC from "./malabr_js/svc.js";

const MODEL_REGISTRY = {
  logistic: LogisticRegression,
  // random_forest: RandomForestClassifier,
  // svc: SVC,
};

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForReady(model, options = {}) {
  const maxAttempts = options.maxAttempts || 20;
  const intervalMs = options.intervalMs || 500;
  const clientId = options.clientId;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    const status = await model.check_status(clientId);
    if (status && status.status === "ready") {
      return status;
    }
    if (status && status.status === "failed") {
      throw new Error(status.error_msg || "Training failed");
    }
    await sleep(intervalMs);
  }

  throw new Error("Model did not become ready in time");
}

async function run() {
  const modelKey = "logistic";
  const clientId = "client_2";
  const ModelClass = MODEL_REGISTRY[modelKey];

  const trainX = [
    -3.0, -2.5, -2.0, -1.5, -1.0,
    -0.8, -0.5, -0.2, 0.2, 0.5,
    0.9, 1.2, 1.6, 2.0, 2.4
  ];
  const trainY = [
    0, 0, 0, 0, 0,
    0, 0, 0, 1, 1,
    1, 1, 1, 1, 1
  ];

  const testX = [-2.4, -1.2, -0.1, 0.3, 1.1, 2.1];
  const testY = [0, 0, 0, 1, 1, 1];

  const model = new ModelClass({}, { name: `${modelKey}_demo_model` });

  try {
    const started = await model.fit(trainX, trainY, clientId);
    const status = await waitForReady(model, { clientId });
    const yPred = await model.predict(testX, clientId);
    const modelScore = await model.score(testX, testY, clientId);

    const checks = {
      fitStarted: started === true,
      statusReady: status && status.status === "ready",
      predictionLengthMatch: Array.isArray(yPred) && yPred.length === testX.length,
      scoreInRange: typeof modelScore === "number" && modelScore >= 0 && modelScore <= 1
    };
    const allPassed = Object.values(checks).every(Boolean);

    console.log("=== MALABR CLASS CLIENT RUN ===");
    console.log("Model details:", model.details());
    console.log("Checks:", checks);
    console.log("All checks passed:", allPassed);
    console.log("Predictions:", yPred);
    console.log("Score:", modelScore);
    console.log("Status:", status);

  } catch (error) {
    console.error("malabr_client_1 run failed:", error.message || error);
  } finally {
    model.destroy();
  }
}

const btn = document.getElementById("run-demo-btn");
const output = document.getElementById("output");

btn.addEventListener("click", async () => {
  output.textContent = "Running demo... Check console for details.";
  run();
});