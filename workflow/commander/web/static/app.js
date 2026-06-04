let sessionId = "";
    let activeInterruptCard = null;
    let activeInterruptSignature = "";
    let currentInterruptValue = null;
    let candidateMenuSuppressed = false;
    let streamInFlight = false;
    let loadingIndicator = null;
    let stateEvents = null;
    const candidateMenu = document.getElementById("candidate-menu");
    const candidateList = document.getElementById("candidate-list");
    const messages = document.getElementById("messages");
    const timeline = document.getElementById("timeline");
    const timelinePanel = timeline.closest("aside");
    const task = document.getElementById("task");
    const form = document.getElementById("chat-form");
    const input = document.getElementById("message-input");
    const sendButton = document.getElementById("send-button");

    function showLoading() {
      if (loadingIndicator) return;
      loadingIndicator = document.createElement("div");
      loadingIndicator.className = "typing-indicator";
      loadingIndicator.setAttribute("aria-label", "等待回覆中");
      loadingIndicator.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
      messages.appendChild(loadingIndicator);
      messages.scrollTop = messages.scrollHeight;
    }

    function hideLoading() {
      if (!loadingIndicator) return;
      loadingIndicator.remove();
      loadingIndicator = null;
    }

    function addMessage(kind, text) {
      hideLoading();
      const div = document.createElement("div");
      div.className = `msg ${kind}`;
      div.textContent = text;
      messages.appendChild(div);
      messages.scrollTop = messages.scrollHeight;
    }

    function scrollTimelineToBottom() {
      requestAnimationFrame(() => {
        timeline.scrollTop = timeline.scrollHeight;
        if (timelinePanel) timelinePanel.scrollTop = timelinePanel.scrollHeight;
      });
    }

    function addNode(event) {
      const div = document.createElement("div");
      div.className = "node";
      div.innerHTML = `<strong>${event.step}. ${event.node_name}</strong><span>${event.status || ""}</span>`;
      timeline.appendChild(div);
      scrollTimelineToBottom();
      const update = event.state_update || {};
      if (event.node_name === "task_classification_node") {
        task.textContent = update.task_intent || "";
      }
      if (event.node_name === "input_node" && update.task) {
        task.textContent = update.task.normalized_task || update.task.original_user_request || "";
      }
    }

    function updateSendButtonState() {
      sendButton.disabled = streamInFlight || currentInterruptValue !== null;
    }

    function parseSseBuffer(buffer, onEvent, flush = false) {
      buffer = buffer.replace(/\r\n/g, "\n");
      const parts = buffer.split("\n\n");
      const rest = parts.pop();
      for (const part of parts) {
        let event = "message";
        let data = "";
        for (const line of part.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) data += line.slice(5).trimStart();
        }
        if (data) onEvent(event, JSON.parse(data));
      }
      if (flush && rest.trim()) {
        let event = "message";
        let data = "";
        for (const line of rest.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) data += line.slice(5).trimStart();
        }
        if (data) onEvent(event, JSON.parse(data));
        return "";
      }
      return rest;
    }

    async function streamPost(url, body) {
      streamInFlight = true;
      updateSendButtonState();
      showLoading();
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      });
      if (!response.ok || !response.body) {
        hideLoading();
        addMessage("system", `HTTP error ${response.status}`);
        streamInFlight = false;
        updateSendButtonState();
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        buffer = parseSseBuffer(buffer, handleEvent);
      }
      buffer += decoder.decode();
      parseSseBuffer(buffer, handleEvent, true);
      hideLoading();
      streamInFlight = false;
      updateSendButtonState();
    }

    async function resumePost(url, body) {
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      });
      if (!response.ok || !response.body) {
        addMessage("system", `HTTP error ${response.status}`);
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        buffer = parseSseBuffer(buffer, handleEvent);
      }
      buffer += decoder.decode();
      parseSseBuffer(buffer, handleEvent, true);
    }

    function handleEvent(type, data) {
      if (type === "node_update") addNode(data);
      if (type === "assistant_message") addMessage("assistant", data.message);
      if (type === "progress_message") {
        addMessage("progress", data.message);
        showLoading();
      }
      if (type === "interrupt") renderInterrupt(data.interrupt, {showChat: true});
      if (type === "done") {
        activeInterruptCard = null;
        activeInterruptSignature = "";
        currentInterruptValue = null;
        renderCandidateMenu(null);
        updateSendButtonState();
        addMessage("system", `Done. Logs: ${data.logs_path}`);
      }
      if (type === "error") addMessage("system", `Error: ${data.error}`);
    }

    function interruptValue(interrupt) {
      return (interrupt && interrupt.interrupts && interrupt.interrupts[0] && interrupt.interrupts[0].value) || {};
    }

    function interruptSignature(interrupt) {
      const value = interruptValue(interrupt);
      const detections = (value.detections || []).map((item) => item.id).join(",");
      const objects = (value.objects || []).map((item) => item.index || item.id || item.label || "").join(",");
      return `${value.type || ""}|${value.message || ""}|d:${detections}|o:${objects}`;
    }

    function localPreviewUrl(previewPath) {
      if (!previewPath) return "";
      if (typeof previewPath === "object") {
        previewPath = previewPath.artifact_path || previewPath.path || "";
      }
      if (!previewPath) return "";
      return `/api/sessions/${sessionId}/preview?path=${encodeURIComponent(String(previewPath))}`;
    }

    function previewUrls(item) {
      const urls = [];
      const addUrl = (url) => {
        if (url && !urls.includes(url)) urls.push(url);
      };
      const addPath = (path) => {
        const url = localPreviewUrl(path);
        if (url) addUrl(url);
      };
      const addMany = (value, handler) => {
        if (Array.isArray(value)) {
          for (const entry of value) handler(entry);
        } else {
          handler(value);
        }
      };
      addMany(item.preview_urls || [], addUrl);
      addUrl(item.preview_url || "");
      addMany(item.preview_paths || [], addPath);
      addPath(item.preview_path);
      return urls;
    }

    function makePreviewFrame(item, url, index) {
      const preview = document.createElement("div");
      preview.className = "preview";
      const image = document.createElement("img");
      const label = item.instance_key || item.label || "candidate";
      image.alt = `${label} preview ${index + 1}`;
      image.onerror = () => {
        image.remove();
        preview.textContent = "Preview missing";
      };
      preview.appendChild(image);
      image.src = url;
      return preview;
    }

    function makePreviewGallery(item) {
      const urls = previewUrls(item);
      const gallery = document.createElement("div");
      gallery.className = `preview-gallery${urls.length > 1 ? " multi" : ""}`;
      if (!urls.length) {
        const preview = document.createElement("div");
        preview.className = "preview";
        preview.textContent = "No preview";
        gallery.appendChild(preview);
        return gallery;
      }
      urls.forEach((url, index) => {
        gallery.appendChild(makePreviewFrame(item, url, index));
      });
      return gallery;
    }

    function renderCandidateMenu(value = currentInterruptValue) {
      candidateList.replaceChildren();
      const detections = (value && value.detections) || [];
      const hasPrompt = Boolean(value && value.type);

      if (detections.length || hasPrompt) {
        const card = document.createElement("div");
        card.className = "interrupt-card";
        fillInterruptCard(card, value);
        candidateList.appendChild(card);
        candidateMenu.hidden = false;
        activeInterruptCard = card;
        return;
      }

      if (candidateMenuSuppressed) {
        candidateMenu.hidden = true;
        return;
      }

      candidateMenu.hidden = true;
    }

    function makeManualInput(value) {
      const row = document.createElement("div");
      row.className = "manual-input";
      const manual = document.createElement("input");
      manual.type = "text";
      manual.placeholder = value.type === "detection_selection"
        ? "輸入編號或 no"
        : "輸入回覆";
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "送出";
      const submit = () => {
        const text = manual.value.trim();
        if (!text) return;
        submitInterruptChoice({value: text}, button);
      };
      button.onclick = submit;
      manual.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          submit();
        }
      });
      row.appendChild(manual);
      row.appendChild(button);
      requestAnimationFrame(() => manual.focus());
      return row;
    }

    function makeChoiceButton(item, mode) {
      const button = document.createElement("button");
      button.type = "button";
      if (mode === "detection") {
        button.className = "choice detection-card";
        button.appendChild(makePreviewGallery(item));

        const detail = document.createElement("div");
        const title = document.createElement("div");
        title.className = "choice-title";
        title.textContent = `[${item.id}] ${item.instance_key || item.label || "target"}`;
        const meta = document.createElement("div");
        meta.className = "choice-meta";
        const cameraText = (item.target_camera_names || item.camsrc || []).join(", ") || "camera unknown";
        meta.textContent = `${cameraText} · center=${JSON.stringify(item.center_world || [])}`;
        detail.appendChild(title);
        detail.appendChild(meta);
        button.appendChild(detail);
        button.onclick = () => submitInterruptChoice({selected_detection_id: item.id}, button);
        return button;
      }

      button.className = "choice";
      button.textContent = "No valid target";
      button.onclick = () => submitInterruptChoice({selected_detection_id: 0}, button);
      return button;
    }

    function fillInterruptCard(card, value) {
      const title = document.createElement("div");
      title.className = "interrupt-title";
      title.textContent = value.message || "Input required";
      card.appendChild(title);

      const detections = value.detections || [];

      if (value.type === "legacy_input") {
        card.appendChild(makeManualInput(value));
        return;
      }

      if (value.type === "detection_selection" && !detections.length) {
        card.appendChild(makeManualInput(value));
      }

      const grid = document.createElement("div");
      grid.className = "choice-grid";
      card.appendChild(grid);

      for (const item of detections) grid.appendChild(makeChoiceButton(item, "detection"));
      grid.appendChild(makeChoiceButton({}, "none"));

      if (!detections.length) {
        const note = document.createElement("div");
        note.className = "empty-choice-note";
        note.textContent = "目前沒有可顯示的候選項。";
        card.insertBefore(note, grid);
      }
    }

    function submitInterruptChoice(payload, button) {
      const card = button ? button.closest(".interrupt-card") : activeInterruptCard;
      if (!card) {
        resumePost(`/api/sessions/${sessionId}/resume/stream`, payload);
        return;
      }
	      const buttons = card.querySelectorAll("button");
      for (const item of buttons) {
        item.disabled = true;
        if (item !== button) item.classList.add("faded");
      }
      if (button) button.classList.add("selected");
      candidateMenuSuppressed = true;
      setTimeout(() => {
        card.classList.add("closing");
      }, 120);
      setTimeout(() => {
        const message = card.closest(".interrupt-message");
        if (message) {
          message.remove();
        } else {
          card.remove();
        }
        if (!message && activeInterruptCard) {
          const activeMessage = activeInterruptCard.closest(".interrupt-message");
          if (activeMessage) activeMessage.remove();
          activeInterruptCard = null;
        }
        if (activeInterruptCard === card) {
          activeInterruptCard = null;
          activeInterruptSignature = "";
        }
        currentInterruptValue = null;
        candidateMenuSuppressed = true;
        renderCandidateMenu(null);
        updateSendButtonState();
      }, 320);
      resumePost(`/api/sessions/${sessionId}/resume/stream`, payload);
    }

    function renderInterrupt(interrupt, options = {}) {
      hideLoading();
      const value = interruptValue(interrupt);
      const signature = interruptSignature(interrupt);
      candidateMenuSuppressed = false;
      currentInterruptValue = value;
      renderCandidateMenu(value);
      updateSendButtonState();
      if (options.showChat === false) return;
      if (activeInterruptSignature === signature) {
        messages.scrollTop = messages.scrollHeight;
        return;
      }
      activeInterruptSignature = signature;
      addMessage("assistant", value.message || "請在左側候選照片中選擇目標。");
      messages.scrollTop = messages.scrollHeight;
    }

    function syncState(data, options = {}) {
	      const state = data.state || {};
      if (state.task) task.textContent = state.task.normalized_task || state.task.original_user_request || state.task_label || task.textContent;
      const hasPending = Object.prototype.hasOwnProperty.call(data, "pending_interrupt");
      const pending = hasPending ? data.pending_interrupt : (state.pending_interrupt || null);
      if (pending && pending.interrupts && pending.interrupts.length) {
        renderInterrupt(pending, {showChat: options.showChat !== false});
      } else {
        currentInterruptValue = null;
        activeInterruptCard = null;
        activeInterruptSignature = "";
        renderCandidateMenu(null);
        updateSendButtonState();
      }
    }

    function connectStateEvents() {
      if (!sessionId || typeof EventSource === "undefined") return;
      if (stateEvents) stateEvents.close();
      stateEvents = new EventSource(`/api/sessions/${sessionId}/events`);
      stateEvents.addEventListener("state", (event) => {
        syncState(JSON.parse(event.data), {showChat: true});
      });
      stateEvents.onerror = () => {
      };
    }

    window.addEventListener("beforeunload", () => {
      if (stateEvents) stateEvents.close();
    });

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      addMessage("user", text);
      streamPost(`/api/sessions/${sessionId}/messages/stream`, {message: text});
    });

	    async function restoreSession(savedSessionId) {
	      if (!savedSessionId) return false;
	      try {
	        const response = await fetch(`/api/sessions/${savedSessionId}/state`);
	        if (!response.ok) return false;
	        const data = await response.json();
	        sessionId = savedSessionId;
	        document.getElementById("session").textContent = sessionId;
        addMessage("system", `Restored session. Logs: ${data.logs_path}`);
        syncState(data, {showChat: true});
        connectStateEvents();
        return true;
	      } catch (error) {
	        return false;
	      }
	    }

	    async function init() {
	      const savedSessionId = localStorage.getItem("vlmRlSessionId") || "";
	      if (await restoreSession(savedSessionId)) return;
	      const response = await fetch("/api/sessions", {
	        method: "POST",
	        headers: {"Content-Type": "application/json"},
	        body: JSON.stringify({})
	      });
	      const data = await response.json();
	      sessionId = data.session_id;
	      localStorage.setItem("vlmRlSessionId", sessionId);
	      document.getElementById("session").textContent = sessionId;
      addMessage("assistant", data.greeting);
      addMessage("system", `Logs: ${data.logs_path}`);
      connectStateEvents();
    }
	    init();
