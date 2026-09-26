const emailInput = document.getElementById("emailInput");
const checkButton = document.getElementById("checkButton");
const buttonText = document.getElementById("buttonText");
const charCount = document.getElementById("charCount");
const errorBox = document.getElementById("errorBox");
const resultBox = document.getElementById("resultBox");
const resultIcon = document.getElementById("resultIcon");
const resultTitle = document.getElementById("resultTitle");
const resultDescription = document.getElementById("resultDescription");
const confidenceValue = document.getElementById("confidenceValue");
const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");

emailInput.addEventListener("input", () => {
  charCount.textContent = emailInput.value.length.toLocaleString();
});

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

function setLoading(loading) {
  checkButton.disabled = loading;
  buttonText.textContent = loading ? "Analyzing..." : "Check Email";
}

async function loadStats() {
  try {
    const response = await fetch("/api/stats");
    if (!response.ok) throw new Error("Model is not ready yet.");
    const stats = await response.json();

    document.getElementById("records").textContent = stats.records.toLocaleString();
    document.getElementById("features").textContent = stats.features.toLocaleString();
    document.getElementById("accuracy").textContent = stats.accuracy + "%";

    statusPill.classList.add("ready");
    statusText.textContent = "Model ready";
  } catch {
    statusText.textContent = "Starting model...";
    setTimeout(loadStats, 2500);
  }
}

async function checkEmail() {
  clearError();
  resultBox.classList.add("hidden");

  const email = emailInput.value.trim();

  if (!email) {
    showError("Paste an email message before running the classifier.");
    emailInput.focus();
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "The classifier could not process this email.");
    }

    const spam = data.is_spam;

    resultBox.className = "result " + (spam ? "spam" : "safe");
    resultIcon.textContent = spam ? "🚨" : "✓";
    resultTitle.textContent = data.prediction;
    resultDescription.textContent = spam
      ? "The model found patterns associated with spam messages."
      : "The model found patterns more consistent with a legitimate message.";
    confidenceValue.textContent = data.confidence.toFixed(2) + "%";
  } catch (error) {
    showError(error.message || "Unable to reach the prediction server.");
  } finally {
    setLoading(false);
  }
}

checkButton.addEventListener("click", checkEmail);

emailInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    checkEmail();
  }
});

loadStats();
