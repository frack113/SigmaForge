(() => {
	const TOAST_TTL = 3500;

	function escHtml(s) {
		return String(s)
			.replace(/&/g, "&amp;")
			.replace(/</g, "&lt;")
			.replace(/>/g, "&gt;")
			.replace(/"/g, "&quot;");
	}

	function createContainer() {
		let c = document.getElementById("toast-container");
		if (!c) {
			c = document.createElement("div");
			c.id = "toast-container";
			c.className = "toast-container";
			document.body.appendChild(c);
		}
		return c;
	}

	window.showToast = function showToast(message, type) {
		if (!type) type = "info";
		const container = createContainer();
		const toast = document.createElement("div");
		toast.className = `toast ${type}`;
		toast.textContent = message;
		container.appendChild(toast);
		setTimeout(() => toast.remove(), TOAST_TTL);
	};

	window.showConfirm = function showConfirm(message) {
		return new Promise((resolve) => {
			const overlay = document.createElement("div");
			overlay.className = "modal";
			overlay.style.display = "flex";

			const box = document.createElement("div");
			box.className = "modal-content confirm-dialog";

			box.innerHTML = `
				<p style="margin:0 0 20px;font-size:var(--text-base);color:var(--text-body);line-height:1.5">${escHtml(message)}</p>
				<div style="display:flex;gap:8px;justify-content:flex-end">
					<button class="btn btn-secondary btn-sm confirm-cancel">Cancel</button>
					<button class="btn btn-danger btn-sm confirm-ok">Confirm</button>
				</div>
			`;

			overlay.appendChild(box);
			document.body.appendChild(overlay);

			function close(result) {
				overlay.remove();
				resolve(result);
			}

			overlay.addEventListener("click", (e) => {
				if (e.target === overlay) close(false);
			});

			box
				.querySelector(".confirm-cancel")
				.addEventListener("click", () => close(false));
			box
				.querySelector(".confirm-ok")
				.addEventListener("click", () => close(true));

			overlay.addEventListener("keydown", (e) => {
				if (e.key === "Escape") close(false);
				if (e.key === "Enter") close(true);
			});

			box.querySelector(".confirm-ok").focus();
		});
	};
})();
