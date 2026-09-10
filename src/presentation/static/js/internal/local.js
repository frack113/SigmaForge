// Local files management

const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const localFilesBody = document.getElementById('local-files-body');
const totalCount = document.getElementById('total-count');
const pendingCount = document.getElementById('pending-count');
const embeddedCount = document.getElementById('embedded-count');

// Drag and drop handlers
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('keydown', (e) => {
	if (e.key === 'Enter' || e.key === ' ') {
		e.preventDefault();
		fileInput.click();
	}
});

uploadZone.addEventListener('dragover', (e) => {
	e.preventDefault();
	uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
	uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
	e.preventDefault();
	uploadZone.classList.remove('dragover');
	if (e.dataTransfer.files.length > 0) {
		handleFiles(e.dataTransfer.files);
	}
});

fileInput.addEventListener('change', (e) => {
	if (e.target.files.length > 0) {
		handleFiles(e.target.files);
		e.target.value = '';
	}
});

async function formatFileSize(bytes) {
	if (bytes === 0) return '0 B';
	const k = 1024;
	const sizes = ['B', 'KB', 'MB', 'GB'];
	const i = Math.floor(Math.log(bytes) / Math.log(k));
	return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function getFileType(contentType) {
	if (!contentType || contentType === 'unknown') return '<span class="status-badge discovery">Unknown</span>';
	return contentType.charAt(0).toUpperCase() + contentType.slice(1);
}

function getEmbedStatusBadge(embedStatus) {
	const statusMap = {
		'discovery': { class: 'discovery', label: 'Pending' },
		'embedded': { class: 'embedded', label: 'Embedded' },
		'error': { class: 'error', label: 'Error' },
	};
	const config = statusMap[embedStatus] || { class: 'discovery', label: embedStatus || 'Unknown' };
	return `<span class="status-badge ${config.class}">${config.label}</span>`;
}

function formatTimestamp(isoStr) {
	if (!isoStr) return '-';
	const d = new Date(isoStr);
	return d.toLocaleDateString('fr-FR') + ' ' + d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
}

async function loadLocalFiles() {
	try {
		const response = await fetch('/api/v1/files/local/list?limit=1000&offset=0');
		const data = await response.json();

		if (!data.success || !data.data) {
			localFilesBody.innerHTML = '<tr><td colspan="5" class="empty-state">No local files found.</td></tr>';
			totalCount.textContent = '0';
			pendingCount.textContent = '0';
			embeddedCount.textContent = '0';
			return;
		}

		const files = data.data;
		totalCount.textContent = data.total || files.length;
		pendingCount.textContent = files.filter(f => f.embed_status === 'discovery').length;
		embeddedCount.textContent = files.filter(f => f.embed_status === 'embedded').length;

		if (files.length === 0) {
			localFilesBody.innerHTML = '<tr><td colspan="5" class="empty-state">No local files found. Upload a file to get started.</td></tr>';
			return;
		}

		localFilesBody.innerHTML = files.map((file, idx) => `
			<tr>
				<td>${file.file_name || file.title || '?'}</td>
				<td>${getFileType(file.content_type)}</td>
				<td>${file.file_size ? formatFileSize(file.file_size) : '0 B'}</td>
				<td>${getEmbedStatusBadge(file.embed_status)}</td>
				<td><button class="btn btn-danger btn-sm" onclick="deleteFile(${idx}, '${file.original_url}')">Delete</button></td>
			</tr>
		`).join('');

	} catch (err) {
		console.error('Failed to load local files:', err);
		localFilesBody.innerHTML = '<tr><td colspan="5" class="empty-state">Error loading files.</td></tr>';
	}
}

async function deleteFile(idx, originalUrl) {
	if (!(await showConfirm('Are you sure you want to delete this file? This cannot be undone.'))) return;

	try {
		const response = await fetch(`/api/v1/files/local/delete?file_path=${encodeURIComponent(originalUrl)}`);
		const data = await response.json();

		if (data.success) {
			showToast('File deleted');
			loadLocalFiles();
		} else {
			showToast('Error: ' + (data.error || 'Unknown error'), 'error');
		}
	} catch (err) {
		console.error('Failed to delete file:', err);
		showToast('Error deleting file.', 'error');
	}
}

async function handleFiles(fileList) {
	const uploadProgress = document.getElementById('upload-progress');
	const statusMessage = document.getElementById('status-message');
	const pb = new ProgressBar({
		container: uploadProgress,
		fill: document.getElementById('progress-fill'),
		text: document.getElementById('status-message'),
	});

	pb.show();
	pb.setText(`Uploading ${fileList.length} file(s)...`;

	let successCount = 0;
	let errorCount = 0;

	for (let i = 0; i < fileList.length; i++) {
		const file = fileList[i];
		pb.setProgress(((i + 1) / fileList.length) * 100);
		pb.setText(`Uploading ${i + 1}/${fileList.length}: ${file.name}`);

		try {
			const formData = new FormData();
			formData.append('file', file);
			formData.append('collection_name', 'local');

			const response = await fetch('/api/v1/files/local/add', {
				method: 'POST',
				body: formData,
			});

			const data = await response.json();
			if (data.success) {
				successCount++;
			} else {
				errorCount++;
				console.error('Upload failed:', data.error);
			}
		} catch (err) {
			errorCount++;
			console.error('Upload error:', err);
		}
	}

	pb.complete();
	pb.setText(`Complete: ${successCount} uploaded, ${errorCount} errors`);

	setTimeout(() => {
		pb.hide();
		loadLocalFiles();
	}, 2000);
}

// Load files on page load
loadLocalFiles();
setInterval(loadLocalFiles, 30000);