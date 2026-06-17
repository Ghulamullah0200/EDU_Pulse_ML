// ── API Configuration ────────────────────────────────────────────
// Uses /api prefix — works both locally (Vite proxy) and on Vercel (serverless)
const API_URL = "/api";

// ── DOM Elements ─────────────────────────────────────────────────
const authShell   = document.getElementById('auth-shell');
const appShell    = document.getElementById('app-shell');
const loginForm   = document.getElementById('login-form');
const authError   = document.getElementById('auth-error');
const pageContainer = document.getElementById('page-container');
const navItems    = document.querySelectorAll('.nav-item');
const logoutBtn   = document.getElementById('logout-btn');

// Modal elements
const modalOverlay = document.getElementById('modal-overlay');
const modalBox     = document.getElementById('modal-box');
const modalTitle   = document.getElementById('modal-title');
const modalBody    = document.getElementById('modal-body');
const modalClose   = document.getElementById('modal-close');

// ── Global State ─────────────────────────────────────────────────
let charts = {};

// ── Modal Helpers ────────────────────────────────────────────────
function openModal(title, bodyHTML) {
    modalTitle.textContent = title;
    modalBody.innerHTML = bodyHTML;
    modalOverlay.classList.remove('hidden');
}

function closeModal() {
    modalOverlay.classList.add('hidden');
    modalBody.innerHTML = '';
}

modalClose.addEventListener('click', closeModal);
modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) closeModal();
});

// ── Authentication ───────────────────────────────────────────────
function checkAuth() {
    const isLoggedIn = localStorage.getItem('edupulse_auth') === 'true';
    if (isLoggedIn) {
        authShell.classList.add('hidden');
        appShell.classList.remove('hidden');
        handleRoute();
    } else {
        authShell.classList.remove('hidden');
        appShell.classList.add('hidden');
    }
}

loginForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const email = document.getElementById('email').value.trim().toLowerCase();
    const pwd   = document.getElementById('password').value;

    // Only allow the authorized user
    if (email === 'ghulam@edupulse.site' && pwd === 'bahzad12') {
        localStorage.setItem('edupulse_auth', 'true');
        authError.classList.add('hidden');
        checkAuth();
    } else {
        authError.textContent = 'Invalid email or password.';
        authError.classList.remove('hidden');
    }
});

logoutBtn.addEventListener('click', () => {
    localStorage.removeItem('edupulse_auth');
    checkAuth();
});

// ── Routing ──────────────────────────────────────────────────────
window.addEventListener('hashchange', handleRoute);

function handleRoute() {
    let hash = window.location.hash || '#/dashboard';
    const route = hash.replace('#', '');

    navItems.forEach(item => {
        item.classList.toggle('active', item.getAttribute('data-route') === route);
    });

    pageContainer.innerHTML = '';

    if (route === '/dashboard')       renderDashboard();
    else if (route === '/predict')    renderPredict();
    else if (route === '/students')   renderStudents();
    else if (route === '/dataset')    renderDataset();
    else if (route === '/model') {
        const tpl = document.getElementById('tpl-model').content.cloneNode(true);
        pageContainer.appendChild(tpl);
    } else {
        pageContainer.innerHTML = `<div class="page"><h2>Coming Soon</h2><p>The ${route} module is under development.</p></div>`;
    }
}


// ══════════════════════════════════════════════════════════════════
//  MODULE: Dashboard
// ══════════════════════════════════════════════════════════════════
function renderDashboard() {
    const tpl = document.getElementById('tpl-dashboard').content.cloneNode(true);
    pageContainer.appendChild(tpl);

    fetch(`${API_URL}/dashboard/summary`)
        .then(r => r.json())
        .then(data => initCharts(data))
        .catch(() => initCharts({}));
}

function initCharts() {
    Object.values(charts).forEach(c => c.destroy());

    const riskCtx    = document.getElementById('chart-risk');
    const deptCtx    = document.getElementById('chart-dept');
    const scatterCtx = document.getElementById('chart-scatter');

    if (riskCtx) {
        charts.risk = new Chart(riskCtx, {
            type: 'doughnut',
            data: {
                labels: ['Not At Risk', 'At Risk'],
                datasets: [{ data: [10080, 1920], backgroundColor: ['#146c2e', '#ba1a1a'] }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }
    if (deptCtx) {
        charts.dept = new Chart(deptCtx, {
            type: 'bar',
            data: {
                labels: ['Engineering', 'Social Sciences', 'Business', 'Computer Science'],
                datasets: [{ label: 'High Risk Students', data: [120, 150, 90, 90], backgroundColor: '#005ac1' }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
    }
    if (scatterCtx) {
        charts.scatter = new Chart(scatterCtx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Students',
                    data: Array.from({ length: 50 }, () => ({
                        x: Math.random() * 60 + 40,
                        y: Math.random() * 60 + 40
                    })),
                    backgroundColor: '#565e71'
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Attendance %' } },
                    y: { title: { display: true, text: 'Final Score' } }
                }
            }
        });
    }
}


// ══════════════════════════════════════════════════════════════════
//  MODULE: Student Directory
// ══════════════════════════════════════════════════════════════════
function renderStudents() {
    const tpl = document.getElementById('tpl-students').content.cloneNode(true);
    pageContainer.appendChild(tpl);

    // Wire enroll button
    document.getElementById('btn-enroll-student').addEventListener('click', showEnrollModal);

    // Fetch real students from the backend
    loadStudentTable();
}

async function loadStudentTable() {
    const tbody = document.getElementById('student-table-body');
    try {
        const res = await fetch(`${API_URL}/students`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        const students = json.data;

        tbody.innerHTML = '';
        students.forEach(s => {
            const risk = s.risk_label || 'Pending';
            let badgeClass = 'badge';
            if (risk === 'At Risk')     badgeClass = 'badge error';
            else if (risk === 'Not At Risk') badgeClass = 'badge success';
            else                        badgeClass = 'badge warning';

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${s.student_id}</strong></td>
                <td>${s.department}</td>
                <td>${s.semester}</td>
                <td>${(s.previous_gpa ?? 0).toFixed(2)}</td>
                <td>${(s.attendance_percentage ?? 0).toFixed(0)}%</td>
                <td><span class="${badgeClass}">${risk}</span></td>
                <td><button class="btn btn-secondary btn-view-profile" data-sid="${s.student_id}" style="padding:6px 12px;font-size:12px;">View Profile</button></td>
            `;
            tbody.appendChild(tr);
        });

        // Attach click handlers for view profile buttons
        tbody.querySelectorAll('.btn-view-profile').forEach(btn => {
            btn.addEventListener('click', () => viewStudentProfile(btn.dataset.sid));
        });
    } catch (err) {
        console.error('Student load error:', err);
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:40px;color:#ba1a1a;">Failed to load students. Is the backend running on port 8000?</td></tr>`;
    }
}

// ── View Profile Modal ───────────────────────────────────────────
async function viewStudentProfile(studentId) {
    openModal('Student Profile', '<p style="text-align:center;padding:40px;color:#978F66;">Loading…</p>');
    try {
        const res = await fetch(`${API_URL}/students/${studentId}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const s = await res.json();

        const risk = s.risk_label || 'Pending';
        let badgeClass = 'badge';
        if (risk === 'At Risk')          badgeClass = 'badge error';
        else if (risk === 'Not At Risk') badgeClass = 'badge success';
        else                             badgeClass = 'badge warning';

        modalBody.innerHTML = `
            <div class="profile-grid">
                <div class="profile-section">
                    <h4>Identity</h4>
                    <div class="profile-row"><span class="profile-label">Student ID</span><span class="profile-value">${s.student_id}</span></div>
                    <div class="profile-row"><span class="profile-label">Age</span><span class="profile-value">${s.age}</span></div>
                    <div class="profile-row"><span class="profile-label">Gender</span><span class="profile-value">${s.gender}</span></div>
                    <div class="profile-row"><span class="profile-label">Department</span><span class="profile-value">${s.department}</span></div>
                    <div class="profile-row"><span class="profile-label">Semester</span><span class="profile-value">${s.semester}</span></div>
                </div>
                <div class="profile-section">
                    <h4>Academics</h4>
                    <div class="profile-row"><span class="profile-label">Previous GPA</span><span class="profile-value">${(s.previous_gpa ?? 0).toFixed(2)}</span></div>
                    <div class="profile-row"><span class="profile-label">Attendance</span><span class="profile-value">${(s.attendance_percentage ?? 0).toFixed(1)}%</span></div>
                    <div class="profile-row"><span class="profile-label">Study Hrs/Wk</span><span class="profile-value">${(s.study_hours_per_week ?? 0).toFixed(1)}</span></div>
                    <div class="profile-row"><span class="profile-label">Assignment Avg</span><span class="profile-value">${(s.assignment_average ?? 0).toFixed(1)}</span></div>
                    <div class="profile-row"><span class="profile-label">Midterm Score</span><span class="profile-value">${(s.midterm_score ?? 0).toFixed(1)}</span></div>
                    <div class="profile-row"><span class="profile-label">Final Score</span><span class="profile-value">${(s.final_score ?? 0).toFixed(1)}</span></div>
                    <div class="profile-row"><span class="profile-label">Absences</span><span class="profile-value">${s.absences ?? 0}</span></div>
                </div>
                <div class="profile-section">
                    <h4>Additional Info</h4>
                    <div class="profile-row"><span class="profile-label">Internet Access</span><span class="profile-value">${s.internet_access}</span></div>
                    <div class="profile-row"><span class="profile-label">Part-Time Job</span><span class="profile-value">${s.part_time_job}</span></div>
                    <div class="profile-row"><span class="profile-label">Extracurricular</span><span class="profile-value">${(s.extracurricular_hours_per_week ?? 0).toFixed(1)} hrs/wk</span></div>
                    <div class="profile-row"><span class="profile-label">Extra Support</span><span class="profile-value">${s.extra_academic_support}</span></div>
                    <div class="profile-row"><span class="profile-label">Data Origin</span><span class="profile-value">${s.data_origin}</span></div>
                </div>
            </div>
            <div style="margin-top:24px;padding:16px;background:var(--smart-header-bg);border-radius:8px;display:flex;justify-content:space-between;align-items:center;">
                <span style="font-weight:700;font-family:var(--font-outfit);color:var(--smart-primary);">Risk Assessment</span>
                <span class="${badgeClass}" style="font-size:14px;">${risk}</span>
            </div>
        `;
    } catch (err) {
        modalBody.innerHTML = `<p style="color:#ba1a1a;text-align:center;padding:20px;">Failed to load profile for ${studentId}.<br><small>${err.message}</small></p>`;
    }
}

// ── Enroll Student Modal ─────────────────────────────────────────
function showEnrollModal() {
    openModal('Enroll New Student', `
        <form id="enroll-form">
            <div class="form-grid">
                <div class="input-group"><label>Full Name</label><input type="text" id="e-name" required placeholder="John Doe"></div>
                <div class="input-group"><label>Age</label><input type="number" id="e-age" min="15" max="40" value="20" required></div>
                <div class="input-group"><label>Gender</label><select id="e-gender"><option>Male</option><option>Female</option></select></div>
                <div class="input-group"><label>Department</label><select id="e-dept"><option>Engineering</option><option>Computer Science</option><option>Business</option><option>Social Sciences</option></select></div>
                <div class="input-group"><label>Semester</label><input type="number" id="e-sem" min="1" max="10" value="1" required></div>
                <div class="input-group"><label>Previous GPA</label><input type="number" id="e-gpa" step="0.01" max="4.0" value="0.0" required></div>
                <div class="input-group"><label>Attendance %</label><input type="number" id="e-attend" step="0.1" max="100" value="0" required></div>
                <div class="input-group"><label>Study Hrs/Wk</label><input type="number" id="e-study" step="0.1" value="0" required></div>
            </div>
            <div class="mt-4">
                <button type="submit" class="btn btn-primary btn-block" id="btn-enroll-submit">Enroll Student</button>
            </div>
        </form>
    `);

    document.getElementById('enroll-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('btn-enroll-submit');
        btn.textContent = 'Enrolling…';
        btn.disabled = true;

        const payload = {
            name: document.getElementById('e-name').value,
            age: parseInt(document.getElementById('e-age').value),
            gender: document.getElementById('e-gender').value,
            department: document.getElementById('e-dept').value,
            semester: parseInt(document.getElementById('e-sem').value),
            previous_gpa: parseFloat(document.getElementById('e-gpa').value),
            attendance_percentage: parseFloat(document.getElementById('e-attend').value),
            study_hours_per_week: parseFloat(document.getElementById('e-study').value),
        };

        try {
            const res = await fetch(`${API_URL}/students/enroll`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const result = await res.json();
            closeModal();
            // Refresh the student table to show the new record in real-time
            loadStudentTable();
        } catch (err) {
            btn.textContent = 'Enroll Student';
            btn.disabled = false;
            alert('Enrollment failed: ' + err.message);
        }
    });
}


// ══════════════════════════════════════════════════════════════════
//  MODULE: Predict Risk
// ══════════════════════════════════════════════════════════════════
function renderPredict() {
    const tpl = document.getElementById('tpl-predict').content.cloneNode(true);
    pageContainer.appendChild(tpl);

    document.getElementById('prediction-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('btn-submit-predict');
        btn.textContent = 'Analyzing…';
        btn.disabled = true;

        const payload = {
            age: parseInt(document.getElementById('p-age').value),
            gender: document.getElementById('p-gender').value,
            department: document.getElementById('p-department').value,
            semester: parseInt(document.getElementById('p-semester').value),
            study_hours_per_week: parseFloat(document.getElementById('p-study').value),
            attendance_percentage: parseFloat(document.getElementById('p-attendance').value),
            assignment_average: parseFloat(document.getElementById('p-assignment').value),
            midterm_score: parseFloat(document.getElementById('p-midterm').value),
            previous_gpa: parseFloat(document.getElementById('p-gpa').value),
            absences: parseInt(document.getElementById('p-absences').value),
            internet_access: document.getElementById('p-internet').value,
            extracurricular_hours_per_week: parseFloat(document.getElementById('p-extra').value),
            extra_academic_support: document.getElementById('p-support').value,
            part_time_job: document.getElementById('p-job').value
        };

        try {
            const res = await fetch(`${API_URL}/predict/single`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            showPredictionResult(data);
        } catch (error) {
            console.error(error);
            alert('Prediction failed. Is the backend running on port 8000?');
        } finally {
            btn.textContent = 'Execute AI Prediction';
            btn.disabled = false;
        }
    });
}

function showPredictionResult(data) {
    const resCard     = document.getElementById('prediction-result');
    const badge       = document.getElementById('res-badge');
    const prob        = document.getElementById('res-prob');
    const factorsList = document.getElementById('res-factors');
    const intervention = document.getElementById('res-intervention');
    const version     = document.getElementById('res-version');

    resCard.classList.remove('hidden');

    badge.textContent = data.prediction;
    badge.className = 'badge ' + (data.risk_level === 'High' ? 'error' : data.risk_level === 'Medium' ? 'warning' : 'success');

    prob.textContent = (data.probability * 100).toFixed(1) + '%';

    factorsList.innerHTML = '';
    data.important_factors.forEach(f => {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${f.factor}:</strong> ${f.message}`;
        factorsList.appendChild(li);
    });

    intervention.textContent = data.recommended_intervention;
    version.textContent = data.model_version;
}


// ══════════════════════════════════════════════════════════════════
//  MODULE: Dataset Explorer
// ══════════════════════════════════════════════════════════════════
let datasetSkip  = 0;
const datasetLimit = 100;
let datasetTotal = 0;

function renderDataset() {
    const tpl = document.getElementById('tpl-dataset').content.cloneNode(true);
    pageContainer.appendChild(tpl);

    datasetSkip = 0;
    document.getElementById('dataset-table-body').innerHTML = '';
    document.getElementById('btn-load-more').addEventListener('click', loadMoreDataset);

    loadMoreDataset();
}

async function loadMoreDataset() {
    const btn = document.getElementById('btn-load-more');
    btn.disabled = true;
    btn.textContent = 'Loading…';

    try {
        const res = await fetch(`${API_URL}/dataset?skip=${datasetSkip}&limit=${datasetLimit}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        const rows = json.data;
        datasetTotal = json.total;

        // Update KPI
        const kpiTotalEl = document.getElementById('kpi-total');
        if (kpiTotalEl) kpiTotalEl.textContent = datasetTotal.toLocaleString();

        const totalCountEl = document.getElementById('total-count');
        if (totalCountEl) totalCountEl.textContent = datasetTotal.toLocaleString();

        const tbody = document.getElementById('dataset-table-body');

        rows.forEach(row => {
            const badgeClass = row.data_origin === 'Original' ? 'badge' : 'badge warning';
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${row.student_id}</strong></td>
                <td>${row.age ?? '—'}</td>
                <td>${row.gender ? row.gender.charAt(0) : '—'}</td>
                <td>${row.department ? row.department.substring(0, 3) : '—'}</td>
                <td>${row.semester ?? '—'}</td>
                <td>${row.study_hours_per_week != null ? row.study_hours_per_week.toFixed(1) : '—'}</td>
                <td>${row.attendance_percentage != null ? row.attendance_percentage.toFixed(0) + '%' : '—'}</td>
                <td>${row.assignment_average != null ? row.assignment_average.toFixed(0) : '—'}</td>
                <td>${row.midterm_score != null ? row.midterm_score.toFixed(0) : '—'}</td>
                <td>${row.previous_gpa != null ? row.previous_gpa.toFixed(2) : '—'}</td>
                <td>${row.absences ?? '—'}</td>
                <td><span class="${badgeClass}">${row.data_origin ?? '—'}</span></td>
            `;
            tbody.appendChild(tr);
        });

        datasetSkip += rows.length;
        document.getElementById('loaded-count').textContent = datasetSkip;

        if (datasetSkip >= datasetTotal) {
            btn.textContent = 'All Records Loaded ✓';
            btn.disabled = true;
        } else {
            btn.textContent = `Load More Records (${datasetSkip} / ${datasetTotal.toLocaleString()})`;
            btn.disabled = false;
        }
    } catch (err) {
        console.error('Dataset load error:', err);
        btn.textContent = 'Retry Loading';
        btn.disabled = false;
        const tbody = document.getElementById('dataset-table-body');
        if (tbody && tbody.children.length === 0) {
            tbody.innerHTML = `<tr><td colspan="12" style="text-align:center;padding:40px;color:#ba1a1a;">Failed to load dataset. Is the backend running on port 8000?</td></tr>`;
        }
    }
}


// ── Start App ────────────────────────────────────────────────────
checkAuth();
