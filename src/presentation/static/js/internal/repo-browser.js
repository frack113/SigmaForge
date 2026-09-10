class RepoBrowser {
	constructor(config) {
		this.apiBase = config.apiBase;
		this.treeEl = config.treeEl;
		this.countEl = config.countEl;
		this.secondaryBtnId = config.secondaryBtnId;
		this.secondaryBtnLabel = config.secondaryBtnLabel;
		this.discoveryWorkers = config.discoveryWorkers;
		this.useSelectedEndpoint = config.useSelectedEndpoint ?? false;
		this.onSecondaryClick = config.onSecondaryClick || (async () => {});
		this.repoCardClass = config.repoCardClass || "repo-card";
		this.secondaryWorkerIndex =
			config.secondaryWorkerIndex !== undefined
				? config.secondaryWorkerIndex
				: 2;
		this.secondaryProgressActiveText =
			config.secondaryProgressActiveText || "References...";
		this.secondaryProgressPctText =
			config.secondaryProgressPctText || "References {pct}%..";

		this.currentRepo = null;
		this.detailContainer = null;
		this._discoveryPollInterval = null;

		this._init();
	}

	formatDate(isoString) {
		if (!isoString) return "Unknown";
		try {
			const date = new Date(isoString);
			return date.toLocaleDateString() + " " + date.toLocaleTimeString();
		} catch {
			return "Unknown";
		}
	}

	setButtonText(id, text) {
		const btn = document.getElementById(id);
		if (btn) btn.textContent = text;
	}

	setButtonsDisabled(disabled) {
		const btn = document.getElementById(this.secondaryBtnId);
		if (btn) btn.disabled = disabled;
	}

	async loadRepos() {
		try {
			const response = await fetch(this.apiBase + "/repos");
			if (!response.ok) throw new Error("API error");
			const data = await response.json();
			const repos = Array.isArray(data) ? data : data.repos || [];
			const container = document.getElementById("repos-list");
			const prompt = document.getElementById("select-repo-prompt");
			if (!container) return;

			if (repos.length === 0) {
				container.innerHTML = "<p>No repositories managed</p>";
				if (prompt) prompt.style.display = "none";
				return;
			}

			if (prompt) prompt.style.display = "block";

			container.innerHTML = repos
				.map((repo) =>
					[
						'<div class="' + this.repoCardClass + '">',
						"  <h3>" + escHtml(repo.org) + "/" + escHtml(repo.name) + "</h3>",
						"  <p>Last commit: " +
							escHtml(this.formatDate(repo.last_commit)) +
							"</p>",
						'  <div class="repo-actions">',
						'    <button data-action="sync" data-org="' +
							escAttr(repo.org) +
							'" data-name="' +
							escAttr(repo.name) +
							'" class="btn btn-sm ' +
							escAttr(repo.sync_class || "btn-unknown") +
							'">Sync</button>',
						'    <button data-action="browse" data-org="' +
							escAttr(repo.org) +
							'" data-name="' +
							escAttr(repo.name) +
							'" class="btn btn-sm btn-secondary">Browse</button>',
						'    <button data-action="delete" data-org="' +
							escAttr(repo.org) +
							'" data-name="' +
							escAttr(repo.name) +
							'" class="btn btn-danger btn-sm">Delete</button>',
						"  </div>",
						"</div>",
					].join("\n"),
				)
				.join("\n");
		} catch (error) {
			console.error("Failed to load repos:", error);
			const container = document.getElementById("repos-list");
			if (container) container.innerHTML = "<p>Error loading repositories</p>";
		}
	}

	async syncRepo(org, name) {
		if (!(await showConfirm("Sync repository " + org + "/" + name + "?")))
			return;
		try {
			const response = await fetch(
				this.apiBase + "/repos/" + org + "/" + name + "/sync",
				{
					method: "POST",
					headers: { "Content-Type": "application/json" },
				},
			);
			const result = await response.json();
			if (result.success) {
				showToast("Syncing repository...");
				this.loadRepos();
			} else {
				showToast(result.error || "Failed to sync", "error");
			}
		} catch {
			showToast("Failed to sync repository", "error");
		}
	}

	async deleteRepo(org, name) {
		if (!(await showConfirm("Delete repository " + org + "/" + name + "?")))
			return;
		try {
			const response = await fetch(
				this.apiBase + "/repos/" + org + "/" + name,
				{ method: "DELETE" },
			);
			const result = await response.json();
			if (result.success) {
				if (
					this.currentRepo &&
					this.currentRepo.org === org &&
					this.currentRepo.name === name
				) {
					this.closeRepoDetail();
				}
				this.loadRepos();
			} else {
				showToast(result.error || "Failed to delete", "error");
			}
		} catch {
			showToast("Failed to delete repository", "error");
		}
	}

	async openRepoDetail(org, name) {
		this.currentRepo = { org: org, name: name };
		document.getElementById("selected-repo-name").textContent =
			org + "/" + name;

		if (this.detailContainer) {
			this.detailContainer.destroy();
			this.detailContainer = null;
		}

		const tree = document.getElementById(this.treeEl);
		tree.innerHTML = '<p class="dir-browser-empty">Loading directories...</p>';

		document.getElementById("repo-detail").style.display = "block";
		document.getElementById(this.countEl).textContent = "";

		const endpoints = {
			list: this.apiBase + "/repos/" + org + "/" + name + "/tree",
			save: this.apiBase + "/repos/" + org + "/" + name + "/select-dirs",
		};
		if (this.useSelectedEndpoint) {
			endpoints.selected =
				this.apiBase + "/repos/" + org + "/" + name + "/selected-dirs";
		}

		this.detailContainer = new DirBrowser(this.treeEl, {
			mode: "tree",
			endpoints: endpoints,
			saveMethod: "POST",
			saveOnChange: true,
			selectedCountEl: this.countEl,
		});
		this.detailContainer.load();
	}

	async saveRepoDetail() {
		const org = this.currentRepo?.org;
		const name = this.currentRepo?.name;
		if (!org || !name) return;

		if (this.detailContainer) {
			await this.detailContainer.save();
		}

		const btn = document.getElementById("save-detail-btn");
		const origLabel = btn?.textContent || "Save";
		if (btn) {
			btn.textContent = "Scanning...";
			btn.disabled = true;
		}

		try {
			const response = await fetch(
				this.apiBase + "/repos/" + org + "/" + name + "/scan",
				{ method: "POST" },
			);
			const result = await response.json();
			if (result.success) {
				this.startDiscoveryPolling();
			} else if (btn) {
				btn.textContent = origLabel;
				btn.disabled = false;
			}
		} catch {
			if (btn) {
				btn.textContent = origLabel;
				btn.disabled = false;
			}
		}
	}

	closeRepoDetail() {
		if (this.detailContainer) {
			this.detailContainer.destroy();
			this.detailContainer = null;
		}
		document.getElementById("repo-detail").style.display = "none";

		this.currentRepo = null;

		document.getElementById(this.treeEl).innerHTML =
			'<p class="dir-browser-empty">Loading directories...</p>';
	}

	closeModal() {
		const modal = document.getElementById("add-repo-modal");
		if (modal) {
			modal.style.display = "none";
			modal.setAttribute("aria-hidden", "true");
			const triggerBtn = document.getElementById("add-repo-btn");
			if (triggerBtn) triggerBtn.focus();
			document.getElementById("add-repo-form").reset();
		}
	}

	isWorkerActive(status) {
		return status === "running" || status === "waiting";
	}

	async pollWorkerStatus(workerType) {
		try {
			const resp = await fetch("/api/v1/dispatcher/status/" + workerType);
			if (!resp.ok) return null;
			return await resp.json();
		} catch {
			return null;
		}
	}

	async checkAnyWorkerActive() {
		const results = await Promise.all(
			this.discoveryWorkers.map((w) => this.pollWorkerStatus(w)),
		);
		return results.some((s) => s && this.isWorkerActive(s.status));
	}

	stopDiscoveryPolling() {
		if (this._discoveryPollInterval) {
			clearInterval(this._discoveryPollInterval);
			this._discoveryPollInterval = null;
		}
	}

	startDiscoveryPolling() {
		this.stopDiscoveryPolling();
		this.setButtonsDisabled(true);
		this._discoveryPollInterval = setInterval(async () => {
			const results = await Promise.all(
				this.discoveryWorkers.map((w) => this.pollWorkerStatus(w)),
			);

			const anyActive = results.some((s) => s && this.isWorkerActive(s.status));

			if (!anyActive) {
				this.stopDiscoveryPolling();
				this.setButtonsDisabled(false);
				this.setButtonText("save-detail-btn", "Saved");
				this.setButtonText(this.secondaryBtnId, this.secondaryBtnLabel);
				setTimeout(() => {
					this.setButtonText("save-detail-btn", "Save");
					const btn = document.getElementById("save-detail-btn");
					if (btn) btn.disabled = false;
				}, 2000);
				this.loadRepos();
				this.closeRepoDetail();
				return;
			}

			// secondary worker progress
			if (
				this.secondaryWorkerIndex >= 0 &&
				this.secondaryWorkerIndex < results.length
			) {
				const secResult = results[this.secondaryWorkerIndex];
				if (secResult && this.isWorkerActive(secResult.status)) {
					const pct =
						secResult.progress_percent > 0
							? Math.round(secResult.progress_percent)
							: undefined;
					if (pct !== undefined) {
						this.setButtonText(
							this.secondaryBtnId,
							this.secondaryProgressPctText.replace("{pct}", pct),
						);
					} else {
						this.setButtonText(
							this.secondaryBtnId,
							this.secondaryProgressActiveText,
						);
					}
				} else {
					this.setButtonText(this.secondaryBtnId, this.secondaryBtnLabel);
				}
			}
		}, 2000);
	}

	async checkRunningWorkersOnLoad() {
		const active = await this.checkAnyWorkerActive();
		if (active) {
			this.startDiscoveryPolling();
		}
	}

	async handleSecondaryClick() {
		try {
			const result = await this.onSecondaryClick();
			if (result?.task_id) {
				this.startDiscoveryPolling();
				return;
			}
			if (result?.indexed !== undefined) {
				const btn = document.getElementById(this.secondaryBtnId);
				if (btn) {
					btn.textContent = `Indexed ${result.indexed} docs`;
					btn.disabled = true;
					setTimeout(() => {
						btn.textContent = this.secondaryBtnLabel;
						btn.disabled = false;
					}, 3000);
				}
				return;
			}
		} catch {
			// fall through — reset button
		}
		this.setButtonText(this.secondaryBtnId, this.secondaryBtnLabel);
	}

	async handleAddRepoForm(e) {
		e.preventDefault();
		const url = document.getElementById("repo-url").value.trim();
		const branch =
			document.getElementById("repo-branch").value.trim() || "main";

		if (!url) {
			showToast("Please enter a repository", "error");
			return;
		}

		let repoUrl = url;
		if (!url.startsWith("http") && !url.endsWith(".git")) {
			repoUrl = "https://github.com/" + url + ".git";
		} else if (!url.endsWith(".git")) {
			repoUrl = url + ".git";
		}

		try {
			const response = await fetch(this.apiBase + "/repos", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ url: repoUrl, branch: branch }),
			});
			const result = await response.json();
			if (result.success) {
				this.closeModal();
				this.loadRepos();
			} else {
				showToast(result.error || "Failed to add repository", "error");
			}
		} catch {
			showToast("Failed to add repository", "error");
		}
	}

	_init() {
		const byId = (id) => document.getElementById(id);
		const on = (id, event, fn) => {
			const el = byId(id);
			if (el) el.addEventListener(event, fn);
		};

		// repo list delegated click
		on("repos-list", "click", (e) => {
			const btn = e.target.closest("button[data-action]");
			if (!btn) return;
			const org = btn.dataset.org;
			const name = btn.dataset.name;
			switch (btn.dataset.action) {
				case "sync":
					this.syncRepo(org, name);
					break;
				case "browse":
					this.openRepoDetail(org, name);
					break;
				case "delete":
					this.deleteRepo(org, name);
					break;
			}
		});

		// add-repo modal
		on("add-repo-btn", "click", () => {
			const modal = byId("add-repo-modal");
			if (modal) {
				modal.style.display = "block";
				modal.setAttribute("aria-hidden", "false");
				// Trap focus inside modal
				const firstFocusable = modal.querySelector('input[type="text"]');
				if (firstFocusable) {
					firstFocusable.focus();
					firstFocusable.select();
				}
				// Close on Escape
				const handleKeydown = (e) => {
					if (e.key === "Escape") {
						modal.style.display = "none";
						modal.setAttribute("aria-hidden", "true");
						const triggerBtn = document.getElementById("add-repo-btn");
						if (triggerBtn) triggerBtn.focus();
						document.removeEventListener("keydown", handleKeydown);
					}
				};
				document.addEventListener("keydown", handleKeydown);
			}
		});

		on("modal-cancel-btn", "click", () => this.closeModal());

		// detail save
		on("save-detail-btn", "click", () => this.saveRepoDetail());

		// detail close
		on("close-detail-btn", "click", () => this.closeRepoDetail());

		// sync all
		const self = this;
		on("sync-all-btn", "click", async function () {
			this.disabled = true;
			this.textContent = "Syncing...";
			try {
				const response = await fetch(self.apiBase + "/repos/sync-all", {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
					},
				});
				const result = await response.json();
				if (result.success) {
					showToast("Sync started for all repositories...");
					self.loadRepos();
				} else {
					showToast(result.error || "Failed to sync all", "error");
				}
			} catch {
				showToast("Failed to sync all repositories", "error");
			} finally {
				this.disabled = false;
				this.textContent = "Sync All";
			}
		});

		// refresh
		on("refresh-btn", "click", () => this.loadRepos());

		// secondary action
		on(this.secondaryBtnId, "click", () => this.handleSecondaryClick());

		// add-repo form
		on("add-repo-form", "submit", (e) => this.handleAddRepoForm(e));

		this.loadRepos();
		this.checkRunningWorkersOnLoad();
	}
}
