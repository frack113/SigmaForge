(() => {
	const STORAGE_KEY = "sigmaforge-theme";

	function getTheme() {
		return localStorage.getItem(STORAGE_KEY) || "light";
	}

	function setTheme(theme) {
		document.documentElement.setAttribute("data-theme", theme);
		localStorage.setItem(STORAGE_KEY, theme);
		const checkbox = document.getElementById("theme-toggle-checkbox");
		if (checkbox) {
			checkbox.checked = theme === "dark";
		}
	}

	function toggleTheme() {
		const next = getTheme() === "dark" ? "light" : "dark";
		setTheme(next);
	}

	document.addEventListener("DOMContentLoaded", () => {
		setTheme(getTheme());
		const checkbox = document.getElementById("theme-toggle-checkbox");
		if (checkbox) {
			checkbox.addEventListener("change", toggleTheme);
		}
	});
})();
