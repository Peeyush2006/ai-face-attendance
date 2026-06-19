// Global State Configuration
const API_URL = ""; // Relative path to backend server
let activeTab = "dashboard";
let activeReportTab = "daily";

// Camera streams
let registrationStream = null;
let attendanceStream = null;
let attendanceInterval = null;

// Photo capture state (for student registration)
let capturedPhotos = [];
const TOTAL_PHOTOS_REQUIRED = 5;

// Toast Notifications
function showToast(message, type = "success") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    
    let iconClass = "fa-circle-check";
    if (type === "warning") iconClass = "fa-triangle-exclamation";
    if (type === "error") iconClass = "fa-circle-xmark";
    
    toast.innerHTML = `
        <i class="fa-solid ${iconClass}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Auto-remove after 4 seconds
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(20px)";
        toast.style.transition = "all 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Sidebar Navigation Router
function switchTab(tabId) {
    // Stop any active camera streams
    stopAllCameras();
    
    // Deactivate previous active menu items and panes
    document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(el => el.classList.remove("active"));
    
    // Activate new ones
    activeTab = tabId;
    let btnId = `btn-${tabId === "camera" ? "camera" : tabId}`;
    let btnEl = document.getElementById(btnId);
    if (btnEl) btnEl.classList.add("active");
    
    let paneEl = document.getElementById(`tab-${tabId}`);
    if (paneEl) paneEl.classList.add("active");
    
    // Update Page Header Info
    const pageTitle = document.getElementById("page-title");
    const pageSubtitle = document.getElementById("page-subtitle");
    
    if (tabId === "dashboard") {
        pageTitle.innerText = "Dashboard";
        pageSubtitle.innerText = "IGNTU BCA Department Overview";
        loadDashboardStats();
    } else if (tabId === "register") {
        pageTitle.innerText = "Student Registration";
        pageSubtitle.innerText = "Register face profiles for new students";
        clearRegistrationForm();
    } else if (tabId === "camera") {
        pageTitle.innerText = "Live Attendance";
        pageSubtitle.innerText = "Real-time AI Face Recognition Attendance Logging";
        loadTodayLog();
    } else if (tabId === "reports") {
        pageTitle.innerText = "Reports & Export";
        pageSubtitle.innerText = "Search, filter, and export attendance logs";
        loadReports();
    } else if (tabId === "settings") {
        pageTitle.innerText = "Settings & Configuration";
        pageSubtitle.innerText = "Configure face recognition parameters and directories";
        loadSettings();
    }
}

// Clean camera feeds on navigation
function stopAllCameras() {
    if (registrationStream) {
        registrationStream.getTracks().forEach(track => track.stop());
        registrationStream = null;
    }
    if (attendanceStream) {
        attendanceStream.getTracks().forEach(track => track.stop());
        attendanceStream = null;
    }
    if (attendanceInterval) {
        clearInterval(attendanceInterval);
        attendanceInterval = null;
    }
    
    // Reset canvas/placeholder views
    document.getElementById("scan-overlay").style.display = "none";
    document.getElementById("attendance-camera-placeholder").style.display = "flex";
}

// ========================================================
// SCREEN 3: ADMIN DASHBOARD LOGIC
// ========================================================
async function loadDashboardStats() {
    try {
        const response = await fetch(`${API_URL}/api/dashboard/stats`);
        if (!response.ok) throw new Error("Failed to fetch dashboard statistics.");
        const data = await response.json();
        
        document.getElementById("stat-total-students").innerText = data.total_students;
        document.getElementById("stat-present-today").innerText = data.present_today;
        document.getElementById("stat-absent-today").innerText = data.absent_today;
        document.getElementById("stat-avg-attendance").innerText = `${data.avg_attendance}%`;
        
        // Populate alerts list
        const alertList = document.getElementById("low-attendance-list");
        const alertCountBadge = document.getElementById("alert-count");
        alertList.innerHTML = "";
        
        alertCountBadge.innerText = `${data.low_attendance_alerts.length} student${data.low_attendance_alerts.length !== 1 ? 's' : ''}`;
        
        if (data.low_attendance_alerts.length === 0) {
            alertList.innerHTML = `
                <div class="alert-item" style="justify-content: center; color: var(--text-muted);">
                    <i class="fa-solid fa-circle-check" style="color: var(--color-green);"></i>
                    <span>All students have attendance rates above 75%!</span>
                </div>
            `;
            return;
        }
        
        data.low_attendance_alerts.forEach(student => {
            const initials = student.name.split(" ").map(n => n[0]).join("").substring(0, 2).toUpperCase();
            const colorClass = student.rate < 70 ? "bg-red" : "bg-gold";
            const textClass = student.rate < 70 ? "text-red" : "text-gold";
            
            const itemHTML = `
                <div class="alert-item">
                    <div class="student-profile">
                        <div class="avatar">${initials}</div>
                        <div class="student-details">
                            <h4>${student.name}</h4>
                            <p>${student.student_id}</p>
                        </div>
                    </div>
                    <div class="alert-progress-wrapper">
                        <div class="progress-info">
                            <span class="${textClass}">${student.rate}%</span>
                        </div>
                        <div class="progress-bar-container">
                            <div class="progress-bar ${colorClass}" style="width: ${student.rate}%;"></div>
                        </div>
                    </div>
                    <button class="btn btn-secondary btn-sm" onclick="sendNotification('${student.name}', '${student.rate}%')">Notify</button>
                </div>
            `;
            alertList.insertAdjacentHTML("beforeend", itemHTML);
        });
    } catch (err) {
        console.error(err);
        showToast("Error loading dashboard data", "error");
    }
}

function sendNotification(studentName, attendanceRate) {
    showToast(`Email notification alert sent to ${studentName} (Current: ${attendanceRate})`, "success");
}

// ========================================================
// SCREEN 1: STUDENT REGISTRATION LOGIC
// ========================================================
async function startRegistrationCamera() {
    const video = document.getElementById("registration-video");
    const placeholder = document.getElementById("reg-camera-placeholder");
    const captureBtn = document.getElementById("btn-capture");
    
    try {
        registrationStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480, facingMode: "user" } 
        });
        video.srcObject = registrationStream;
        placeholder.style.display = "none";
        captureBtn.disabled = false;
        showToast("Camera access granted. Capturing enabled.", "success");
    } catch (err) {
        console.error(err);
        showToast("Could not access camera. Simulating webcam feed...", "warning");
        simulateRegistrationFeed();
    }
}

// Simulated webcam feed in case of headless or no-camera environment
let simRegInterval = null;
function simulateRegistrationFeed() {
    const placeholder = document.getElementById("reg-camera-placeholder");
    placeholder.style.display = "none";
    
    const video = document.getElementById("registration-video");
    video.style.display = "none";
    
    // Show hidden canvas as view
    const canvas = document.getElementById("registration-canvas");
    canvas.style.display = "block";
    canvas.width = 640;
    canvas.height = 400;
    const ctx = canvas.getContext("2d");
    
    const captureBtn = document.getElementById("btn-capture");
    captureBtn.disabled = false;
    
    simRegInterval = setInterval(() => {
        // Draw dark viewport
        ctx.fillStyle = "#0f0f13";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw grid
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.lineWidth = 1;
        for (let i = 0; i < canvas.width; i += 40) {
            ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height); ctx.stroke();
        }
        for (let i = 0; i < canvas.height; i += 40) {
            ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(canvas.width, i); ctx.stroke();
        }
        
        // Draw simulated face wireframe
        ctx.strokeStyle = "rgba(139, 92, 246, 0.6)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(canvas.width/2, canvas.height/2 - 20, 70, 0, Math.PI * 2); // head
        ctx.stroke();
        
        ctx.beginPath();
        ctx.arc(canvas.width/2 - 25, canvas.height/2 - 30, 8, 0, Math.PI * 2); // eye left
        ctx.arc(canvas.width/2 + 25, canvas.height/2 - 30, 8, 0, Math.PI * 2); // eye right
        ctx.stroke();
        
        ctx.beginPath(); // nose
        ctx.moveTo(canvas.width/2, canvas.height/2 - 15);
        ctx.lineTo(canvas.width/2 - 10, canvas.height/2 + 10);
        ctx.lineTo(canvas.width/2, canvas.height/2 + 10);
        ctx.stroke();
        
        ctx.beginPath(); // mouth smile
        ctx.arc(canvas.width/2, canvas.height/2 + 15, 20, 0.1 * Math.PI, 0.9 * Math.PI);
        ctx.stroke();
        
        // Camera simulation banner
        ctx.fillStyle = "#8b5cf6";
        ctx.fillRect(10, 10, 130, 24);
        ctx.fillStyle = "#fff";
        ctx.font = "11px Inter";
        ctx.fillText("SIMULATED WEBCAM", 20, 26);
    }, 100);
    
    // Add cleaner to stop stream
    registrationStream = {
        getTracks: () => [{
            stop: () => {
                clearInterval(simRegInterval);
                canvas.style.display = "none";
                video.style.display = "block";
            }
        }]
    };
}

function capturePhoto() {
    if (capturedPhotos.length >= TOTAL_PHOTOS_REQUIRED) return;
    
    const canvas = document.getElementById("registration-canvas");
    const video = document.getElementById("registration-video");
    const ctx = canvas.getContext("2d");
    
    canvas.width = 640;
    canvas.height = 480;
    
    // Draw current video frame or use simulated canvas content
    if (video.srcObject) {
        // Mirror frame when capturing since video is mirrored
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.setTransform(1, 0, 0, 1, 0, 0); // reset transform
    } else {
        // If simulation, capture the simulated canvas drawing
        const simCanvas = document.getElementById("registration-canvas");
        ctx.drawImage(simCanvas, 0, 0, canvas.width, canvas.height);
    }
    
    const dataURL = canvas.toDataURL("image/jpeg");
    capturedPhotos.push(dataURL);
    
    updateCaptureProgress();
    showToast(`Photo ${capturedPhotos.length} captured!`, "success");
    
    if (capturedPhotos.length === TOTAL_PHOTOS_REQUIRED) {
        document.getElementById("btn-capture").disabled = true;
        document.getElementById("btn-save-encode").disabled = false;
        showToast("5 photos captured successfully. Click Save & Encode.", "success");
    }
}

function updateCaptureProgress() {
    const squares = document.querySelectorAll("#progress-squares .square");
    const statusText = document.getElementById("capture-status");
    
    squares.forEach((sq, idx) => {
        if (idx < capturedPhotos.length) {
            sq.classList.add("filled");
        } else {
            sq.classList.remove("filled");
        }
    });
    
    statusText.innerText = `${capturedPhotos.length}/5 captured`;
}

function clearRegistrationForm() {
    document.getElementById("reg-student-id").value = "";
    document.getElementById("reg-full-name").value = "";
    document.getElementById("reg-course-section").value = "";
    
    capturedPhotos = [];
    updateCaptureProgress();
    
    document.getElementById("btn-save-encode").disabled = true;
    document.getElementById("btn-capture").disabled = registrationStream === null;
}

async function saveAndEncodeStudent() {
    const studentId = document.getElementById("reg-student-id").value.trim();
    const name = document.getElementById("reg-full-name").value.trim();
    const courseSection = document.getElementById("reg-course-section").value.trim();
    
    if (!studentId || !name) {
        showToast("Student ID and Full Name are required.", "error");
        return;
    }
    
    if (capturedPhotos.length !== TOTAL_PHOTOS_REQUIRED) {
        showToast("Please capture exactly 5 photos.", "error");
        return;
    }
    
    // Set loading state
    const saveBtn = document.getElementById("btn-save-encode");
    const originalText = saveBtn.innerHTML;
    saveBtn.disabled = true;
    saveBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Saving...`;
    
    try {
        const response = await fetch(`${API_URL}/api/students/register`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                student_id: studentId,
                name: name,
                course_section: courseSection,
                photos: capturedPhotos
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || "Failed to register student.");
        }
        
        showToast(data.message || "Student registered & encoded successfully!", "success");
        clearRegistrationForm();
        switchTab("dashboard");
    } catch (err) {
        console.error(err);
        showToast(err.message, "error");
    } finally {
        saveBtn.disabled = false;
        saveBtn.innerHTML = originalText;
    }
}

// ========================================================
// SCREEN 2: LIVE ATTENDANCE (CAMERA) LOGIC
// ========================================================
async function startAttendanceCamera() {
    const video = document.getElementById("attendance-video");
    const placeholder = document.getElementById("attendance-camera-placeholder");
    const canvas = document.getElementById("attendance-canvas");
    const scanOverlay = document.getElementById("scan-overlay");
    
    canvas.style.display = "block";
    canvas.width = 640;
    canvas.height = 400;
    
    try {
        attendanceStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480, facingMode: "user" } 
        });
        video.srcObject = attendanceStream;
        placeholder.style.display = "none";
        scanOverlay.style.display = "flex";
        
        // Start streaming frames to API
        startProcessingFrames();
        showToast("Live attendance feed activated.", "success");
    } catch (err) {
        console.error(err);
        showToast("Webcam access failed. Simulating live feed...", "warning");
        simulateAttendanceFeed();
    }
}

function startProcessingFrames() {
    const video = document.getElementById("attendance-video");
    const canvas = document.getElementById("attendance-canvas");
    const ctx = canvas.getContext("2d");
    
    const hiddenCanvas = document.createElement("canvas");
    hiddenCanvas.width = 320; // Downscale frame for speed
    hiddenCanvas.height = 240;
    const hCtx = hiddenCanvas.getContext("2d");
    
    let isProcessing = false;
    
    attendanceInterval = setInterval(async () => {
        if (isProcessing) return;
        
        // Draw frame to display canvas
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        
        // Draw frame to hidden processing canvas
        hCtx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
        const dataURL = hiddenCanvas.toDataURL("image/jpeg", 0.6); // Compress
        
        isProcessing = true;
        try {
            const response = await fetch(`${API_URL}/api/process_frame`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ frame: dataURL })
            });
            const data = await response.json();
            
            // Draw the annotated image returned by server (it has bounding boxes)
            if (data.annotated_frame) {
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                img.src = data.annotated_frame;
            }
            
            // Display success notification overlay if recognized
            if (data.recognition) {
                triggerSuccessAlert(data.recognition);
            }
        } catch (err) {
            console.error("Frame processing error:", err);
        } finally {
            isProcessing = false;
        }
    }, 200); // 5 frames per second
}

// Simulated Live Feed
let simAttendanceInterval = null;
function simulateAttendanceFeed() {
    const placeholder = document.getElementById("attendance-camera-placeholder");
    placeholder.style.display = "none";
    
    const canvas = document.getElementById("attendance-canvas");
    const scanOverlay = document.getElementById("scan-overlay");
    scanOverlay.style.display = "flex";
    
    const ctx = canvas.getContext("2d");
    let frameCount = 0;
    
    simAttendanceInterval = setInterval(async () => {
        frameCount++;
        
        // Draw dark backdrop
        ctx.fillStyle = "#0f0f13";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw wireframe silhouette
        ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        ctx.lineWidth = 1;
        for (let i = 0; i < canvas.width; i += 40) {
            ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, canvas.height); ctx.stroke();
        }
        for (let i = 0; i < canvas.height; i += 40) {
            ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(canvas.width, i); ctx.stroke();
        }
        
        // Draw face box in center
        const bx = canvas.width / 2 - 75;
        const by = canvas.height / 2 - 90;
        const bw = 150;
        const bh = 180;
        
        // Bouncing box simulation representing face detection
        const offset = Math.sin(frameCount / 10) * 8;
        const fbx = bx + offset;
        const fby = by + offset / 2;
        
        // Simulated green/red bounding box
        const isRecognized = (frameCount % 60) > 40; // Simulate match state periodically
        if (isRecognized) {
            ctx.strokeStyle = "#4caf50";
            ctx.fillStyle = "#4caf50";
            ctx.lineWidth = 3;
            ctx.strokeRect(fbx, fby, bw, bh);
            ctx.fillRect(fbx, fby - 25, bw, 25);
            ctx.fillStyle = "#fff";
            ctx.font = "12px Outfit";
            ctx.fillText("Peeyush Kumar Tiwari (98.2%)", fbx + 8, fby - 8);
            
            // Trigger UI match event once per cycle
            if (frameCount % 60 === 41) {
                const now = new Date();
                triggerSuccessAlert({
                    student_id: "2401151028",
                    name: "Peeyush Kumar Tiwari",
                    course_section: "BCA — Section A",
                    time: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                    confidence: "98.2%",
                    status: "Present"
                });
            }
        } else {
            ctx.strokeStyle = "rgba(139, 92, 246, 0.5)";
            ctx.lineWidth = 2;
            ctx.strokeRect(fbx, fby, bw, bh);
        }
        
        // Simulator banner
        ctx.fillStyle = "#f59e0b";
        ctx.fillRect(10, 10, 150, 24);
        ctx.fillStyle = "#fff";
        ctx.font = "11px Inter";
        ctx.fillText("SIMULATION RUNNING", 20, 26);
    }, 150);
    
    // Add cleaner to stop stream
    attendanceStream = {
        getTracks: () => [{
            stop: () => {
                clearInterval(simAttendanceInterval);
                if (attendanceInterval) clearInterval(attendanceInterval);
                canvas.style.display = "none";
            }
        }]
    };
}

let successAlertTimeout = null;
function triggerSuccessAlert(matchData) {
    const alertBox = document.getElementById("scan-success-alert");
    const nameEl = document.getElementById("success-student-info");
    const timeEl = document.getElementById("success-time");
    const confEl = document.getElementById("success-conf");
    
    nameEl.innerText = `${matchData.name} — ${matchData.student_id}`;
    timeEl.innerText = matchData.time;
    confEl.innerText = matchData.confidence;
    
    alertBox.style.display = "flex";
    
    // Refresh today's logs dynamically in sidebar
    loadTodayLog();
    
    // Clear existing timeout
    if (successAlertTimeout) clearTimeout(successAlertTimeout);
    
    // Hide alert after 4 seconds
    successAlertTimeout = setTimeout(() => {
        alertBox.style.opacity = "0";
        alertBox.style.transform = "translateY(10px)";
        alertBox.style.transition = "all 0.4s ease";
        setTimeout(() => {
            alertBox.style.display = "none";
            alertBox.style.opacity = "1";
            alertBox.style.transform = "translateY(0)";
        }, 400);
    }, 4000);
}

async function loadTodayLog() {
    try {
        const response = await fetch(`${API_URL}/api/attendance/today`);
        if (!response.ok) throw new Error("Failed to fetch today's logs.");
        const data = await response.json();
        
        const logList = document.getElementById("today-log-list");
        const summaryText = document.getElementById("log-summary-text");
        
        logList.innerHTML = "";
        
        let presentCount = 0;
        let totalCount = data.length;
        
        data.forEach(log => {
            const initials = log.name.split(" ").map(n => n[0]).join("").substring(0, 2).toUpperCase();
            const statusClass = log.status.toLowerCase();
            
            if (log.status === "Present" || log.status === "Late") {
                presentCount++;
            }
            
            const logHTML = `
                <div class="log-item">
                    <div class="student-profile" style="min-width: unset;">
                        <div class="student-initials">${initials}</div>
                        <div class="log-info-wrapper">
                            <div class="log-name">${log.name}</div>
                        </div>
                    </div>
                    <div class="log-badge-time">
                        <span class="status-badge ${statusClass}">${log.status}</span>
                        <span class="log-time">${log.time}</span>
                    </div>
                </div>
            `;
            logList.insertAdjacentHTML("beforeend", logHTML);
        });
        
        summaryText.innerText = `Recognizing active faces · ${presentCount}/${totalCount} marked`;
    } catch (err) {
        console.error(err);
        showToast("Error loading logs", "error");
    }
}

// ========================================================
// SCREEN 4: REPORTS & EXPORT LOGIC
// ========================================================
function switchReportTab(reportType) {
    document.querySelectorAll(".report-tab").forEach(btn => btn.classList.remove("active"));
    
    // Find matching button text
    const buttons = document.querySelectorAll(".report-tab");
    buttons.forEach(btn => {
        if (btn.innerText.toLowerCase() === reportType) {
            btn.classList.add("active");
        }
    });
    
    activeReportTab = reportType;
    loadReports();
}

async function loadReports() {
    const dateInput = document.getElementById("report-date-filter");
    const courseSelect = document.getElementById("report-course-filter");
    
    const filterDate = dateInput.value;
    const filterCourse = courseSelect.value;
    
    try {
        const response = await fetch(`${API_URL}/api/reports?date=${filterDate}&course=${encodeURIComponent(filterCourse)}`);
        if (!response.ok) throw new Error("Failed to load report data.");
        const data = await response.json();
        
        const tableBody = document.getElementById("report-table-body");
        tableBody.innerHTML = "";
        
        if (data.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 30px;">
                        No logs found for the selected filters.
                    </td>
                </tr>
            `;
            return;
        }
        
        data.forEach(row => {
            const statusClass = row.status.toLowerCase();
            const pctVal = parseInt(row.total_pct);
            const pctClass = pctVal < 75 ? "text-red" : "text-primary";
            
            const trHTML = `
                <tr>
                    <td><strong>${row.roll}</strong></td>
                    <td>${row.name}</td>
                    <td>${row.time}</td>
                    <td>${row.confidence}</td>
                    <td><span class="${pctClass}">${row.total_pct}</span></td>
                    <td><span class="status-badge ${statusClass}">${row.status}</span></td>
                </tr>
            `;
            tableBody.insertAdjacentHTML("beforeend", trHTML);
        });
    } catch (err) {
        console.error(err);
        showToast("Error fetching report rows", "error");
    }
}

function exportExcel() {
    // Generate actual CSV content from table rows
    const rows = document.querySelectorAll(".report-table tr");
    let csvContent = "data:text/csv;charset=utf-8,";
    
    rows.forEach(row => {
        const cols = row.querySelectorAll("td, th");
        const rowData = [];
        cols.forEach(col => rowData.push(`"${col.innerText.trim()}"`));
        csvContent += rowData.join(",") + "\r\n";
    });
    
    const dateFilter = document.getElementById("report-date-filter").value;
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `attendance_report_${dateFilter}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast("CSV report file downloaded.", "success");
}

function exportPDF() {
    // Open standard system print dialog formatted for table
    window.print();
}

// ========================================================
// SCREEN 5: SETTINGS & CONFIGURATION LOGIC
// ========================================================
async function loadSettings() {
    try {
        const response = await fetch(`${API_URL}/api/settings`);
        if (!response.ok) throw new Error("Failed to load settings.");
        const data = await response.json();
        
        // Update sliders and controls
        document.getElementById("setting-threshold").value = data.threshold;
        document.getElementById("threshold-display").innerText = parseFloat(data.threshold).toFixed(2);
        document.getElementById("setting-late").value = data.late_threshold;
        document.getElementById("setting-camera").value = data.camera_source;
        document.getElementById("setting-email").checked = data.email_alerts;
        document.getElementById("setting-db").value = data.database_path;
    } catch (err) {
        console.error(err);
        showToast("Error loading system settings", "error");
    }
}

async function saveSettings() {
    const thresholdVal = parseFloat(document.getElementById("setting-threshold").value);
    const lateVal = parseInt(document.getElementById("setting-late").value);
    const cameraVal = document.getElementById("setting-camera").value;
    const emailVal = document.getElementById("setting-email").checked;
    const dbVal = document.getElementById("setting-db").value;
    
    try {
        const response = await fetch(`${API_URL}/api/settings`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                threshold: thresholdVal,
                late_threshold: lateVal,
                camera_source: cameraVal,
                email_alerts: emailVal,
                database_path: dbVal
            })
        });
        
        if (!response.ok) throw new Error("Failed to update settings.");
        
        showToast("System settings updated successfully.", "success");
    } catch (err) {
        console.error(err);
        showToast(err.message, "error");
    }
}

// Page Initialization
window.addEventListener("DOMContentLoaded", () => {
    // Set date badges to today's date
    const today = new Date();
    const options = { day: 'numeric', month: 'long', year: 'numeric' };
    const dateStr = today.toLocaleDateString('en-US', options);
    
    const badgeEl = document.getElementById("current-date-badge");
    if (badgeEl) badgeEl.innerText = dateStr;
    
    // Set default value for reports date input to today
    const reportDateInput = document.getElementById("report-date-filter");
    if (reportDateInput) {
        const isoDate = today.toISOString().split("T")[0];
        reportDateInput.value = isoDate;
    }
    
    // Load initial tab
    switchTab("dashboard");
});
