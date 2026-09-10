(() => {
	function init() {
		const messagesEl = document.getElementById("chat-messages");
		const chatForm = document.getElementById("chat-form");
		const input = document.getElementById("message-input");
		const sendBtn = document.getElementById("send-btn");
		let welcome = document.getElementById("chat-welcome");
		const newChatBtn = document.getElementById("new-chat-btn");
		const typingEl = document.getElementById("typing-indicator");
		const promptSelect = document.getElementById("prompt-select");

		if (!messagesEl || !chatForm || !input || !sendBtn) return;

		if (typeof marked !== "undefined") {
			marked.setOptions({ gfm: true, breaks: true });
		}

		const MD_TAGS = [
			"a",
			"b",
			"i",
			"em",
			"strong",
			"code",
			"pre",
			"br",
			"hr",
			"p",
			"blockquote",
			"ul",
			"ol",
			"li",
			"h1",
			"h2",
			"h3",
			"h4",
			"h5",
			"h6",
			"table",
			"thead",
			"tbody",
			"tfoot",
			"tr",
			"th",
			"td",
			"span",
		];
		const MD_ATTRS = ["href", "title", "colspan", "rowspan"];

		if (typeof DOMPurify !== "undefined") {
			DOMPurify.addHook("afterSanitizeAttributes", (node) => {
				if (node.tagName === "A") {
					node.setAttribute("target", "_blank");
					node.setAttribute("rel", "noopener noreferrer");
				}
			});
		}

		function renderMarkdown(text) {
			if (typeof marked !== "undefined") {
				const normalized = text
					.replace(/([^\n])\n*(#{1,6}\s)/g, "$1\n\n$2")
					.replace(/([^\n])\n*(\|)/g, "$1\n\n$2");
				const html = marked.parse(normalized);
				if (typeof DOMPurify !== "undefined") {
					return DOMPurify.sanitize(html, {
						ALLOWED_TAGS: MD_TAGS,
						ALLOWED_ATTR: MD_ATTRS,
					});
				}
				return html;
			}
			const div = document.createElement("div");
			div.textContent = text;
			return div.innerHTML;
		}

		let currentAbort = null;

		function extractThinkContent(raw) {
			const openIdx = raw.indexOf("<think>");
			const closeIdx = raw.indexOf("</think>");

			if (openIdx === -1) {
				// Suppress flash: if raw starts with < but no full tag yet
				const lastOpen = raw.lastIndexOf("<");
				const lastClose = raw.lastIndexOf(">");
				if (lastOpen > lastClose && raw.trim().length < 10) {
					return { thinking: null, answer: "" };
				}
				return { thinking: null, answer: raw };
			}

			if (closeIdx !== -1 && closeIdx > openIdx) {
				return {
					thinking: raw.substring(openIdx + 7, closeIdx),
					answer: raw.substring(closeIdx + 8),
				};
			}

			return {
				thinking: raw.substring(openIdx + 7),
				answer: raw.substring(0, openIdx),
			};
		}

		function populatePromptSelect(prompts) {
			if (!promptSelect) return;
			promptSelect.innerHTML = "";

			if (!prompts || prompts.length === 0) {
				const emptyOpt = document.createElement("option");
				emptyOpt.value = "";
				emptyOpt.textContent = "— No prompts —";
				promptSelect.appendChild(emptyOpt);
				return;
			}

			let activeId = null;
			prompts.forEach((p) => {
				const opt = document.createElement("option");
				opt.value = p.id;
				opt.textContent = p.name;
				promptSelect.appendChild(opt);
				if (p.is_active) {
					activeId = p.id;
				}
			});

			if (activeId) {
				promptSelect.value = activeId;
			} else {
				promptSelect.selectedIndex = 0;
			}
		}

		function loadPrompts() {
			if (window.__INITIAL_PROMPTS && window.__INITIAL_PROMPTS.length > 0) {
				populatePromptSelect(window.__INITIAL_PROMPTS);
				return;
			}
			fetch("/api/v1/admin/prompts")
				.then((r) => r.json())
				.then((prompts) => {
					populatePromptSelect(prompts);
				})
				.catch(() => {
					if (promptSelect) {
						promptSelect.innerHTML = '<option value="">— No prompts —</option>';
					}
				});
		}

		function getSelectedPrompt() {
			if (promptSelect) return promptSelect.value || "";
			return "";
		}

		function showTyping() {
			if (typingEl) typingEl.hidden = false;
		}

		function hideTyping() {
			if (typingEl) typingEl.hidden = true;
		}

		function createMessageHeader(role) {
			const header = document.createElement("div");
			header.className = "message-header";
			const label = document.createElement("span");
			label.textContent = role === "user" ? "You" : "Assistant";
			header.appendChild(label);
			return header;
		}

		function createMessageActions(_role, bodyEl) {
			const actions = document.createElement("div");
			actions.className = "message-actions";

			const copyBtn = document.createElement("button");
			copyBtn.className = "btn btn-ghost btn-sm";
			copyBtn.textContent = "Copy";
			copyBtn.addEventListener("click", () => {
				const text = getAnswerText(bodyEl);
				navigator.clipboard
					.writeText(text)
					.then(() => {
						showToast("Copied!");
					})
					.catch(() => {
						showToast("Copy failed");
					});
			});
			actions.appendChild(copyBtn);
			return actions;
		}

		function getAnswerText(bodyEl) {
			const _thinkBlock = bodyEl.querySelector(".thinking-block");
			const clone = bodyEl.cloneNode(true);
			const thinkClone = clone.querySelector(".thinking-block");
			if (thinkClone) thinkClone.remove();
			return clone.textContent || clone.innerText || "";
		}

		function updateMessageBody(bodyEl, thinking, answer, formatted) {
			bodyEl.innerHTML = "";

			if (thinking !== null) {
				const details = document.createElement("details");
				details.className = "thinking-block";
				const summary = document.createElement("summary");
				summary.textContent = "Thinking";
				details.appendChild(summary);
				const content = document.createElement("div");
				content.className = "thinking-content";
				if (formatted) {
					content.innerHTML = renderMarkdown(thinking);
				} else {
					content.textContent = thinking;
				}
				details.appendChild(content);
				bodyEl.appendChild(details);
			}

			if (answer) {
				const answerDiv = document.createElement("div");
				answerDiv.className = "answer-content";
				if (formatted) {
					answerDiv.innerHTML = renderMarkdown(answer);
				} else {
					answerDiv.textContent = answer;
				}
				bodyEl.appendChild(answerDiv);
			}
		}

		function makeMessageEl(role, bodyContent, append) {
			const div = document.createElement("div");
			div.className = `message ${role}`;

			const header = createMessageHeader(role);
			div.appendChild(header);

			const body = document.createElement("div");
			body.className = "message-body";

			if (role === "user") {
				body.textContent = bodyContent;
			} else {
				const parsed = extractThinkContent(bodyContent);
				updateMessageBody(body, parsed.thinking, parsed.answer, true);
			}
			div.appendChild(body);

			const actions = createMessageActions(role, body);
			div.appendChild(actions);

			if (append !== false) {
				messagesEl.appendChild(div);
				messagesEl.scrollTop = messagesEl.scrollHeight;
			}
			return { el: div, bodyEl: body };
		}

		function setLoading(loading) {
			if (loading) {
				sendBtn.innerHTML =
					'<svg viewBox="0 0 24 24" fill="currentColor" style="width:18px;height:18px"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';
				sendBtn.title = "Stop";
				sendBtn.setAttribute("aria-label", "Stop");
				sendBtn.classList.add("is-loading");
				showTyping();
			} else {
				sendBtn.innerHTML =
					'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:18px;height:18px"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>';
				sendBtn.title = "Send";
				sendBtn.setAttribute("aria-label", "Send");
				sendBtn.classList.remove("is-loading");
				hideTyping();
			}
		}

		function loadHistory() {
			fetch("/api/v1/chat/history")
				.then((resp) => resp.json())
				.then((history) => {
					if (history && history.length > 0) {
						if (welcome) {
							welcome.remove();
							welcome = null;
						}
						history.forEach((msg) => {
							makeMessageEl(msg.role, msg.content, true);
						});
					}
				})
				.catch(() => {});
		}

		function clearChat() {
			fetch("/api/v1/chat/history", { method: "DELETE" })
				.then(() => {
					messagesEl.querySelectorAll(".message").forEach((el) => {
						el.remove();
					});
					if (!welcome) {
						const w = document.createElement("div");
						w.className = "chat-welcome";
						w.innerHTML =
							"<h2>SigmaForge</h2><p>Ask questions about Sigma detection rules</p>";
						messagesEl.appendChild(w);
						welcome = w;
					} else {
						welcome.hidden = false;
					}
					hideTyping();
				})
				.catch(() => {
					showToast("Error clearing history");
				});
		}

		if (newChatBtn) {
			newChatBtn.addEventListener("click", async () => {
				const hasMessages = !!messagesEl.querySelector(".message");
				if (!hasMessages) {
					clearChat();
					return;
				}
				const ok = await showConfirm(
					"Start a new chat? This clears the current history.",
				);
				if (ok) clearChat();
			});
		}

		loadPrompts();
		loadHistory();

		/* ---- Chat submit ---- */
		chatForm.addEventListener("submit", async (e) => {
			e.preventDefault();
			const text = input.value.trim();
			if (!text) return;

			if (currentAbort) {
				currentAbort.abort();
				currentAbort = null;
			}

			makeMessageEl("user", text, true);
			input.value = "";
			input.style.height = "auto";
			setLoading(true);

			if (welcome) {
				welcome.remove();
				welcome = null;
			}

			const promptId = getSelectedPrompt();

			currentAbort = new AbortController();
			let resp;
			try {
				resp = await fetch("/api/v1/chat/message/stream", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						message: text,
						mode: "search",
						model: "",
						prompt_id: promptId,
					}),
					signal: currentAbort.signal,
				});
			} catch (err) {
				if (err.name !== "AbortError") {
					makeMessageEl("error", `Network error: ${err.message}`, true);
				}
				setLoading(false);
				currentAbort = null;
				return;
			}

			if (!resp.ok) {
				makeMessageEl(
					"error",
					`Request failed with status ${resp.status}`,
					true,
				);
				setLoading(false);
				currentAbort = null;
				return;
			}

			const reader = resp.body.getReader();
			const decoder = new TextDecoder();
			let accumulated = "";
			let bubbleInfo = null;
			let gotToken = false;
			let sseBuffer = "";

			try {
				while (true) {
					let result;
					try {
						result = await reader.read();
					} catch (err) {
						if (err.name !== "AbortError") {
							makeMessageEl("error", `Stream read error: ${err.message}`, true);
						}
						break;
					}
					if (result.done) break;

					sseBuffer += decoder.decode(result.value, { stream: true });

					for (
						let eventEnd = sseBuffer.indexOf("\n\n");
						eventEnd !== -1;
						eventEnd = sseBuffer.indexOf("\n\n")
					) {
						const event = sseBuffer.substring(0, eventEnd);
						sseBuffer = sseBuffer.substring(eventEnd + 2);

						if (!event.startsWith("data: ")) continue;

						const data = event.slice(6);

						if (data === "[DONE]") {
							if (!gotToken && !bubbleInfo) {
								makeMessageEl("assistant", "(no response from LLM)", true);
							}
							setLoading(false);

							if (bubbleInfo) {
								const finalParsed = extractThinkContent(accumulated);
								updateMessageBody(
									bubbleInfo.bodyEl,
									finalParsed.thinking,
									finalParsed.answer,
									true,
								);
								const ts = document.createElement("div");
								ts.className = "message-timestamp";
								ts.textContent = "just now";
								bubbleInfo.el.appendChild(ts);
							}

							currentAbort = null;
							return;
						}

						if (data.indexOf("__CITATIONS__:") === 0) {
							const citations = (() => {
								try {
									return JSON.parse(data.slice(15));
								} catch {
									return [];
								}
							})();
							if (citations.length > 0) {
								if (!bubbleInfo) {
									hideTyping();
									bubbleInfo = makeMessageEl("assistant", "", true);
								}
								const existing =
									bubbleInfo.el.querySelector(".citations-block");
								if (existing) existing.remove();
								const block = document.createElement("div");
								block.className = "citations-block";
								const title = document.createElement("div");
								title.className = "citations-title";
								title.textContent = "Sources";
								block.appendChild(title);
								const list = document.createElement("div");
								list.className = "citations-list";
								citations.forEach((c) => {
									const item = document.createElement("div");
									item.className = "citation-item";
									item.textContent = c;
									list.appendChild(item);
								});
								block.appendChild(list);
								bubbleInfo.el.appendChild(block);
							}
							continue;
						}

						accumulated += data;
						gotToken = true;

						if (!bubbleInfo) {
							hideTyping();
							bubbleInfo = makeMessageEl("assistant", "", true);
						}

						const parsed = extractThinkContent(accumulated);
						updateMessageBody(
							bubbleInfo.bodyEl,
							parsed.thinking,
							parsed.answer,
							true,
						);
						messagesEl.scrollTop = messagesEl.scrollHeight;
					}
				}

				if (!gotToken && !bubbleInfo) {
					makeMessageEl("assistant", "(no response from LLM)", true);
				}
			} catch (err) {
				if (err.name !== "AbortError") {
					makeMessageEl("error", `Error: ${err.message}`, true);
				}
			}
			setLoading(false);
			currentAbort = null;
		});

		input.addEventListener("input", function () {
			this.style.height = "auto";
			this.style.height = `${this.scrollHeight}px`;
		});

		input.addEventListener("keydown", (e) => {
			if (e.key === "Enter" && !e.shiftKey) {
				e.preventDefault();
				chatForm.requestSubmit();
			}
		});

		input.focus();
	}

	document.addEventListener("DOMContentLoaded", init);
})();
