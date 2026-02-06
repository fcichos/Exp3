/**
 * Q&A Widget for Experimental Physics 3 Course
 * This widget provides an interactive Q&A interface for Quarto documents
 */

(function () {
  "use strict";

  // Configuration
  const CONFIG = {
    apiUrl: window.QA_API_URL || "http://localhost:8000",
    maxRetries: 3,
    retryDelay: 1000,
    defaultContextSize: 5,
    defaultTemperature: 0.7,
    defaultMaxTokens: 500,
  };

  // Widget styles
  const styles = `
        .qa-widget-container {
            max-width: 800px;
            margin: 2rem auto;
            padding: 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .qa-widget-inner {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
        }

        .qa-widget-title {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .qa-widget-subtitle {
            color: #718096;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }

        .qa-input-group {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .qa-input {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 2px solid #e2e8f0;
            border-radius: 6px;
            font-size: 1rem;
            transition: all 0.3s;
            outline: none;
        }

        .qa-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .qa-input::placeholder {
            color: #a0aec0;
        }

        .qa-submit-btn {
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .qa-submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }

        .qa-submit-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .qa-loading {
            display: none;
            text-align: center;
            padding: 1rem;
        }

        .qa-loading.active {
            display: block;
        }

        .qa-spinner {
            display: inline-block;
            width: 30px;
            height: 30px;
            border: 3px solid #e2e8f0;
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .qa-response {
            display: none;
            margin-top: 1.5rem;
            padding: 1.5rem;
            background: #f7fafc;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .qa-response.active {
            display: block;
            animation: fadeIn 0.5s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .qa-answer {
            color: #2d3748;
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        .qa-sources {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
        }

        .qa-sources-title {
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }

        .qa-source-item {
            padding: 0.5rem;
            margin: 0.25rem 0;
            background: white;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #718096;
            border: 1px solid #e2e8f0;
        }

        .qa-source-title {
            font-weight: 600;
            color: #4a5568;
        }

        .qa-source-score {
            display: inline-block;
            padding: 0.125rem 0.375rem;
            background: #edf2f7;
            border-radius: 3px;
            font-size: 0.75rem;
            margin-left: 0.5rem;
        }

        .qa-error {
            display: none;
            padding: 1rem;
            background: #fed7d7;
            color: #742a2a;
            border-radius: 6px;
            margin-top: 1rem;
        }

        .qa-error.active {
            display: block;
        }

        .qa-suggestions {
            margin-top: 1rem;
            padding: 1rem;
            background: #f0fff4;
            border-radius: 6px;
            border: 1px solid #9ae6b4;
        }

        .qa-suggestions-title {
            font-weight: 600;
            color: #276749;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }

        .qa-suggestion {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            margin: 0.25rem;
            background: white;
            border: 1px solid #9ae6b4;
            border-radius: 20px;
            color: #22543d;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .qa-suggestion:hover {
            background: #9ae6b4;
            color: white;
            transform: translateY(-1px);
        }

        .qa-confidence {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.5rem;
            padding: 0.5rem;
            background: #edf2f7;
            border-radius: 4px;
            font-size: 0.85rem;
        }

        .qa-confidence-bar {
            width: 100px;
            height: 6px;
            background: #e2e8f0;
            border-radius: 3px;
            overflow: hidden;
        }

        .qa-confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
            transition: width 0.5s;
        }
    `;

  // Create and inject styles
  function injectStyles() {
    if (!document.getElementById("qa-widget-styles")) {
      const styleElement = document.createElement("style");
      styleElement.id = "qa-widget-styles";
      styleElement.textContent = styles;
      document.head.appendChild(styleElement);
    }
  }

  // Create the widget HTML
  function createWidgetHTML(containerId) {
    return `
            <div class="qa-widget-container">
                <div class="qa-widget-inner">
                    <div class="qa-widget-title">
                        <span>🤖</span>
                        <span>Ask Your Physics Question</span>
                    </div>
                    <div class="qa-widget-subtitle">
                        Get instant answers from the course materials
                    </div>

                    <div class="qa-input-group">
                        <input
                            type="text"
                            class="qa-input"
                            id="${containerId}-input"
                            placeholder="e.g., What is the principle of geometrical optics?"
                            autocomplete="off"
                        />
                        <button class="qa-submit-btn" id="${containerId}-submit">
                            <span>Ask</span>
                            <span>→</span>
                        </button>
                    </div>

                    <div class="qa-suggestions">
                        <div class="qa-suggestions-title">Try these questions:</div>
                        <span class="qa-suggestion" data-question="What is geometrical optics?">Geometrical Optics</span>
                        <span class="qa-suggestion" data-question="Explain wave interference">Wave Interference</span>
                        <span class="qa-suggestion" data-question="What are Maxwell's equations?">Maxwell's Equations</span>
                        <span class="qa-suggestion" data-question="Describe quantum tunneling">Quantum Tunneling</span>
                    </div>

                    <div class="qa-loading" id="${containerId}-loading">
                        <div class="qa-spinner"></div>
                        <div style="margin-top: 0.5rem; color: #718096;">Searching course materials...</div>
                    </div>

                    <div class="qa-error" id="${containerId}-error"></div>

                    <div class="qa-response" id="${containerId}-response">
                        <div class="qa-answer" id="${containerId}-answer"></div>
                        <div class="qa-confidence" id="${containerId}-confidence">
                            <span>Confidence:</span>
                            <div class="qa-confidence-bar">
                                <div class="qa-confidence-fill" id="${containerId}-confidence-fill"></div>
                            </div>
                            <span id="${containerId}-confidence-text">0%</span>
                        </div>
                        <div class="qa-sources" id="${containerId}-sources">
                            <div class="qa-sources-title">📚 Sources:</div>
                            <div id="${containerId}-sources-list"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
  }

  // API call with retry logic
  async function callAPI(endpoint, data, retries = CONFIG.maxRetries) {
    const url = `${CONFIG.apiUrl}${endpoint}`;

    for (let i = 0; i < retries; i++) {
      try {
        const response = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(data),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
      } catch (error) {
        console.error(`API call failed (attempt ${i + 1}/${retries}):`, error);

        if (i === retries - 1) {
          throw error;
        }

        // Wait before retrying
        await new Promise((resolve) =>
          setTimeout(resolve, CONFIG.retryDelay * (i + 1)),
        );
      }
    }
  }

  // Format answer with highlighting
  function formatAnswer(answer) {
    // Convert markdown-like formatting
    answer = answer.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    answer = answer.replace(/\*(.*?)\*/g, "<em>$1</em>");
    answer = answer.replace(/\n\n/g, "</p><p>");
    answer = answer.replace(/\n/g, "<br>");

    // Wrap in paragraphs if not already
    if (!answer.startsWith("<p>")) {
      answer = "<p>" + answer + "</p>";
    }

    return answer;
  }

  // Initialize widget
  function initializeWidget(containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
      console.error(`Container with ID '${containerId}' not found`);
      return;
    }

    // Inject styles
    injectStyles();

    // Add widget HTML
    container.innerHTML = createWidgetHTML(containerId);

    // Get elements
    const input = document.getElementById(`${containerId}-input`);
    const submitBtn = document.getElementById(`${containerId}-submit`);
    const loading = document.getElementById(`${containerId}-loading`);
    const errorDiv = document.getElementById(`${containerId}-error`);
    const responseDiv = document.getElementById(`${containerId}-response`);
    const answerDiv = document.getElementById(`${containerId}-answer`);
    const sourcesDiv = document.getElementById(`${containerId}-sources-list`);
    const confidenceFill = document.getElementById(
      `${containerId}-confidence-fill`,
    );
    const confidenceText = document.getElementById(
      `${containerId}-confidence-text`,
    );

    // Handle suggestion clicks
    container.querySelectorAll(".qa-suggestion").forEach((suggestion) => {
      suggestion.addEventListener("click", function () {
        input.value = this.dataset.question;
        submitBtn.click();
      });
    });

    // Handle form submission
    async function handleSubmit() {
      const question = input.value.trim();

      if (!question) {
        errorDiv.textContent = "Please enter a question";
        errorDiv.classList.add("active");
        return;
      }

      // Reset UI
      errorDiv.classList.remove("active");
      responseDiv.classList.remove("active");
      loading.classList.add("active");
      submitBtn.disabled = true;

      try {
        // Call API
        const response = await callAPI("/ask", {
          question: question,
          context_size: CONFIG.defaultContextSize,
          temperature: CONFIG.defaultTemperature,
          max_tokens: CONFIG.defaultMaxTokens,
        });

        // Display answer
        answerDiv.innerHTML = formatAnswer(response.answer);

        // Display confidence
        const confidencePercent = Math.round(response.confidence * 100);
        confidenceFill.style.width = `${confidencePercent}%`;
        confidenceText.textContent = `${confidencePercent}%`;

        // Display sources
        sourcesDiv.innerHTML = "";
        if (response.sources && response.sources.length > 0) {
          response.sources.forEach((source) => {
            const sourceItem = document.createElement("div");
            sourceItem.className = "qa-source-item";
            sourceItem.innerHTML = `
                            <span class="qa-source-title">${source.title}</span>
                            <span class="qa-source-score">Score: ${source.score.toFixed(3)}</span>
                        `;
            sourcesDiv.appendChild(sourceItem);
          });
        }

        // Show response
        responseDiv.classList.add("active");
      } catch (error) {
        console.error("Error:", error);
        errorDiv.textContent = `Error: ${error.message}. Please make sure the Q&A server is running.`;
        errorDiv.classList.add("active");
      } finally {
        loading.classList.remove("active");
        submitBtn.disabled = false;
      }
    }

    // Event listeners
    submitBtn.addEventListener("click", handleSubmit);
    input.addEventListener("keypress", function (e) {
      if (e.key === "Enter") {
        handleSubmit();
      }
    });
  }

  // Auto-initialize widgets when DOM is ready
  function autoInitialize() {
    // Look for divs with class 'qa-widget'
    document.querySelectorAll(".qa-widget").forEach((element, index) => {
      if (!element.id) {
        element.id = `qa-widget-${index}`;
      }
      initializeWidget(element.id);
    });
  }

  // Initialize on different loading scenarios
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", autoInitialize);
  } else {
    autoInitialize();
  }

  // Export for manual initialization
  window.QAWidget = {
    init: initializeWidget,
    setAPIUrl: function (url) {
      CONFIG.apiUrl = url;
    },
  };
})();
