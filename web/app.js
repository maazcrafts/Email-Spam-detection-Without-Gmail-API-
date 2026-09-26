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
const gmailButton = document.getElementById("gmailButton");
const scanButton = document.getElementById("scanButton");
const disconnectButton = document.getElementById("disconnectButton");
const gmailState = document.getElementById("gmailState");
const gmailHint = document.getElementById("gmailHint");
const mailList = document.getElementById("mailList");
const scanSummary = document.getElementById("scanSummary");

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
    if (!response.ok) throw new Error();
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

function setGmailConnected(connected, configured) {
  if (!configured) {
    gmailButton.disabled = true;
    gmailButton.textContent = "Gmail not configured";
    gmailState.textContent = "Gmail integration needs setup";
    gmailHint.textContent = "Add the Google OAuth environment variables in Vercel.";
    return;
  }

  gmailButton.disabled = false;
  if (connected) {
    gmailButton.textContent = "Gmail Connected";
    gmailState.textContent = "Gmail is connected";
    gmailHint.textContent = "Read-only access is active. Scan your recent inbox messages.";
    scanButton.disabled = false;
    disconnectButton.classList.remove("hidden");
  } else {
    gmailButton.textContent = "Connect Gmail";
    gmailState.textContent = "Gmail not connected";
    gmailHint.textContent = "Connect your Google account to scan recent inbox messages.";
    scanButton.disabled = true;
    disconnectButton.classList.add("hidden");
  }
}

async function loadGmailStatus() {
  try {
    const response = await fetch("/api/gmail/status");
    const data = await response.json();
    setGmailConnected(data.connected, data.configured);

    const params = new URLSearchParams(window.location.search);
    if (params.get("gmail") === "connected") {
      history.replaceState({}, "", window.location.pathname);
      await scanInbox();
    } else if (params.get("gmail") === "denied") {
      history.replaceState({}, "", window.location.pathname);
      showError("Gmail authorization was cancelled.");
    }
  } catch {
    gmailState.textContent = "Gmail status unavailable";
  }
}

gmailButton.addEventListener("click", () => {
  window.location.href = "/api/gmail/login";
});

disconnectButton.addEventListener("click", async () => {
  await fetch("/api/gmail/disconnect", { method: "POST" });
  setGmailConnected(false, true);
  scanSummary.classList.add("hidden");
  mailList.innerHTML = '<div class="empty-state">Connect Gmail to start scanning your inbox.</div>';
});

function renderMessages(messages) {
  if (!messages.length) {
    mailList.innerHTML = '<div class="empty-state">No inbox messages were returned.</div>';
    return;
  }

  mailList.innerHTML = messages.map((mail) => {
    const spamClass = mail.is_spam ? "mail-spam" : "mail-safe";
    const badge = mail.is_spam ? "SPAM" : "NOT SPAM";
    const date = mail.date ? new Date(mail.date).toLocaleString() : "Unknown date";
    const sender = escapeHtml(mail.from);
    const subject = escapeHtml(mail.subject);
    const snippet = escapeHtml(mail.snippet || "No preview available.");

    return `
      <article class="mail-item ${spamClass}">
        <div class="mail-main">
          <div class="mail-topline">
            <span class="mail-badge">${badge}</span>
            <span class="mail-date">${date}</span>
          </div>
          <h4>${subject}</h4>
          <p class="mail-from">${sender}</p>
          <p class="mail-snippet">${snippet}</p>
        </div>
        <div class="mail-confidence">
          <span>CONFIDENCE</span>
          <strong>${mail.confidence.toFixed(1)}%</strong>
        </div>
      </article>
    `;
  }).join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function scanInbox() {
  clearError();
  scanButton.disabled = true;
  scanButton.textContent = "Scanning...";
  mailList.innerHTML = '<div class="empty-state">Fetching and classifying recent inbox messages...</div>';

  try {
    const response = await fetch("/api/gmail/messages?limit=20");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not read Gmail.");

    document.getElementById("scannedCount").textContent = data.total;
    document.getElementById("spamCount").textContent = data.spam_count;
    document.getElementById("safeCount").textContent = data.safe_count;
    scanSummary.classList.remove("hidden");
    renderMessages(data.messages);
  } catch (error) {
    if (error.message.toLowerCase().includes("authorization") || error.message.toLowerCase().includes("connected")) {
      setGmailConnected(false, true);
    }
    showError(error.message || "Unable to scan Gmail.");
    mailList.innerHTML = '<div class="empty-state">The inbox could not be scanned.</div>';
  } finally {
    scanButton.disabled = false;
    scanButton.innerHTML = 'Scan Inbox <span class="button-arrow">→</span>';
  }
}

scanButton.addEventListener("click", scanInbox);

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
    if (!response.ok) throw new Error(data.detail || "The classifier could not process this email.");

    resultBox.className = "result " + (data.is_spam ? "spam" : "safe");
    resultIcon.textContent = data.is_spam ? "🚨" : "✓";
    resultTitle.textContent = data.prediction;
    resultDescription.textContent = data.is_spam
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
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") checkEmail();
});

loadStats();
loadGmailStatus();
