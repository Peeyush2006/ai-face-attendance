import sqlite3
import os
import datetime
import random
import numpy as np

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'attendance.db')

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        course_section TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS student_photos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        photo_path TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT,
        confidence REAL,
        status TEXT NOT NULL, -- 'Present', 'Late', 'Absent'
        FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
        UNIQUE(student_id, date)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    ''')
    
    # Seed default settings
    default_settings = {
        'threshold': '0.50',
        'late_threshold': '15',
        'camera_source': '0',
        'email_alerts': '1',
        'database_path': DB_FILE
    }
    
    for key, val in default_settings.items():
        cursor.execute('INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)', (key, val))
        
    conn.commit()
    
    # Seed mock data if students table is empty
    cursor.execute('SELECT COUNT(*) FROM students')
    if cursor.fetchone()[0] == 0:
        seed_mock_data(conn)
        
    conn.close()

def seed_mock_data(conn):
    cursor = conn.cursor()
    
    # 1. Add specific students from the PDF enrollment list
    pdf_students = [
        ('2401151001', 'Alok Pati Tiwari', 'BCA — Section A'),
        ('2401151002', 'Aman Marko', 'BCA — Section A'),
        ('2401151003', 'Ankit Kumar', 'BCA — Section A'),
        ('2401151004', 'Ansul Rohani', 'BCA — Section A'),
        ('2401151005', 'Arati Choudhari', 'BCA — Section A'),
        ('2401151006', 'Chandan Kumar', 'BCA — Section A'),
        ('2401151007', 'Divyansh Mishra', 'BCA — Section A'),
        ('2401151008', 'Jishan Moin', 'BCA — Section A'),
        ('2401151009', 'Nishi Shukla', 'BCA — Section A'),
        ('2401151010', 'Radheshyam Suthar', 'BCA — Section A'),
        ('2401151011', 'Rewanchal Shah Dhurvey', 'BCA — Section A'),
        ('2401151012', 'Rishu Kumar', 'BCA — Section A'),
        ('2401151013', 'Ritika Namdev', 'BCA — Section A'),
        ('2401151014', 'Sagar Banawal', 'BCA — Section A'),
        ('2401151015', 'Sanjana Sahu', 'BCA — Section A'),
        ('2401151016', 'Saurabh Kumar Kewat', 'BCA — Section A'),
        ('2401151017', 'Sheetal Kushwaha', 'BCA — Section A'),
        ('2401151018', 'Shivam Mishra', 'BCA — Section A'),
        ('2401151019', 'Shivani Jaisawal', 'BCA — Section A'),
        ('2401151020', 'Sonal Sen', 'BCA — Section A'),
        ('2401151021', 'Tuleshwar Singh', 'BCA — Section A'),
        ('2401151022', 'Vivek Kumar Chandel', 'BCA — Section A'),
        ('2401151023', 'Vivek Yadav', 'BCA — Section A'),
        ('2401151024', 'Sachin Kumar Sharma', 'BCA — Section A'),
        ('2401151025', 'Muskan Yadav', 'BCA — Section A'),
        ('2401151026', 'Vaibhav Mishra', 'BCA — Section A'),
        ('2401151027', 'Pushkar Pandey', 'BCA — Section A'),
        ('2401151028', 'Peeyush Kumar Tiwari', 'BCA — Section A'),
        ('2401151029', 'Yash Vardhan Namdeo', 'BCA — Section A'),
        ('2401151030', 'Ahsas Singh', 'BCA — Section A')
    ]
    
    for sid, name, cs in pdf_students:
        cursor.execute('INSERT INTO students (student_id, name, course_section) VALUES (?, ?, ?)', (sid, name, cs))
        
    conn.commit()
    
    # Get list of all student IDs
    cursor.execute('SELECT student_id, name FROM students')
    all_students = cursor.fetchall()
    
    # We want today's date
    today_str = datetime.date.today().isoformat()
    
    # 2. Seed past 30 days of attendance logs
    # Overall target average is 83%. We can set specific attendance probabilities:
    # Sheetal Kushwaha (2401151017) target is 68%
    # Muskan Yadav (2401151025) target is 72%
    # Others are randomly selected to have mean attendance of ~84%
    
    past_days = 30
    dates = []
    # Generate list of past 30 weekdays (excluding weekends)
    curr = datetime.date.today() - datetime.timedelta(days=past_days)
    while len(dates) < past_days:
        if curr.weekday() < 5:  # Monday to Friday
            dates.append(curr.isoformat())
        curr += datetime.timedelta(days=1)
        
    # Seed historical logs
    for s_row in all_students:
        sid = s_row['student_id']
        name = s_row['name']
        
        # Determine this student's attendance probability
        if sid == '2401151017':  # Sheetal Kushwaha
            p = 0.68
        elif sid == '2401151025':  # Muskan Yadav
            p = 0.72
        elif sid == '2401151028':  # Peeyush Kumar Tiwari
            p = 0.87
        elif sid == '2401151012':  # Rishu Kumar
            p = 0.85
        elif sid == '2401151030':  # Ahsas Singh
            p = 0.80
        elif sid == '2401151026':  # Vaibhav Mishra
            p = 0.90
        else:
            p = random.uniform(0.78, 0.92)  # Average student attendance
            
        # Draw 30 random choices using NumPy (1 = Present/Late, 0 = Absent)
        attendance_draws = np.random.binomial(1, p, len(dates))
        
        for date_str, attended in zip(dates, attendance_draws):
            if attended == 1:
                # Decide if present or late
                is_late = np.random.binomial(1, 0.15)  # 15% chance of being late
                if is_late:
                    status = 'Late'
                    # Late time: e.g. 10:16 to 10:40
                    time_str = f"10:{random.randint(16, 40):02d}"
                else:
                    status = 'Present'
                    # Present time: e.g. 09:50 to 10:15
                    hour = 10 if random.random() > 0.6 else 9
                    minute = random.randint(0, 15) if hour == 10 else random.randint(50, 59)
                    time_str = f"{hour:02d}:{minute:02d}"
                
                cursor.execute('''
                INSERT INTO attendance (student_id, date, time, confidence, status)
                VALUES (?, ?, ?, ?, ?)
                ''', (sid, date_str, time_str, round(random.uniform(90.0, 99.5), 1), status))
            else:
                cursor.execute('''
                INSERT INTO attendance (student_id, date, time, confidence, status)
                VALUES (?, ?, ?, ?, ?)
                ''', (sid, date_str, None, None, 'Absent'))
                
    # 3. Seed today's attendance specifically to match the screenshot:
    # First delete today's attendance if any was generated in the historical loop
    cursor.execute('DELETE FROM attendance WHERE date = ?', (today_str,))
    
    # Specific today records (matches your mockups but with the new IDs)
    today_records = {
        '2401151028': ('Present', '10:14', 98.2),   # Peeyush Kumar Tiwari
        '2401151012': ('Late', '10:32', 94.8),      # Rishu Kumar
        '2401151030': ('Absent', None, None),        # Ahsas Singh
        '2401151026': ('Present', '10:08', 97.4),   # Vaibhav Mishra
        '2401151017': ('Absent', None, None),        # Sheetal Kushwaha (low attendance alert)
        '2401151025': ('Absent', None, None),        # Muskan Yadav (low attendance alert)
    }
    
    for sid, (status, time_str, conf) in today_records.items():
        cursor.execute('''
        INSERT INTO attendance (student_id, date, time, confidence, status)
        VALUES (?, ?, ?, ?, ?)
        ''', (sid, today_str, time_str, conf, status))
        
    # We want exactly 24 present and 6 absent today (Total 30).
    # We already marked:
    # - Present today: 3 (Peeyush, Rishu, Vaibhav)
    # - Absent today: 3 (Ahsas, Sheetal, Muskan)
    # We need to distribute the remaining 24 students:
    # - Present: 24 - 3 = 21
    # - Absent: 6 - 3 = 3
    
    remaining_students = [s for s in all_students if s['student_id'] not in today_records]
    random.shuffle(remaining_students)
    
    # First 21 are present
    for s_row in remaining_students[:21]:
        sid = s_row['student_id']
        is_late = np.random.binomial(1, 0.15)
        if is_late:
            status = 'Late'
            time_str = f"10:{random.randint(16, 45):02d}"
        else:
            status = 'Present'
            hour = 10 if random.random() > 0.6 else 9
            minute = random.randint(0, 15) if hour == 10 else random.randint(45, 59)
            time_str = f"{hour:02d}:{minute:02d}"
        cursor.execute('''
        INSERT INTO attendance (student_id, date, time, confidence, status)
        VALUES (?, ?, ?, ?, ?)
        ''', (sid, today_str, time_str, round(random.uniform(91.0, 99.0), 1), status))
        
    # The remaining 3 are absent
    for s_row in remaining_students[21:]:
        sid = s_row['student_id']
        cursor.execute('''
        INSERT INTO attendance (student_id, date, time, confidence, status)
        VALUES (?, ?, ?, ?, ?)
        ''', (sid, today_str, None, None, 'Absent'))
        
    conn.commit()

# --- Database Access API ---

def get_settings():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT key, value FROM settings')
    settings = {row['key']: row['value'] for row in cursor.fetchall()}
    conn.close()
    return settings

def update_settings(settings_dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    for key, val in settings_dict.items():
        cursor.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, str(val)))
    conn.commit()
    conn.close()

def add_student(student_id, name, course_section):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
        INSERT OR REPLACE INTO students (student_id, name, course_section)
        VALUES (?, ?, ?)
        ''', (student_id, name, course_section))
        conn.commit()
        success = True
    except Exception:
        success = False
    conn.close()
    return success

def add_student_photo(student_id, photo_path):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO student_photos (student_id, photo_path)
    VALUES (?, ?)
    ''', (student_id, photo_path))
    conn.commit()
    conn.close()

def get_student_photos(student_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT photo_path FROM student_photos WHERE student_id = ?', (student_id,))
    photos = [row['photo_path'] for row in cursor.fetchall()]
    conn.close()
    return photos

def get_all_student_photos():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT student_id, photo_path FROM student_photos')
    photos = [{'student_id': row['student_id'], 'photo_path': row['photo_path']} for row in cursor.fetchall()]
    conn.close()
    return photos

def mark_attendance(student_id, status, time_str=None, confidence=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    today_str = datetime.date.today().isoformat()
    if status in ['Present', 'Late'] and time_str is None:
        time_str = datetime.datetime.now().strftime('%H:%M')
        
    try:
        cursor.execute('''
        INSERT OR REPLACE INTO attendance (student_id, date, time, confidence, status)
        VALUES (?, ?, ?, ?, ?)
        ''', (student_id, today_str, time_str, confidence, status))
        conn.commit()
        success = True
    except Exception:
        success = False
    conn.close()
    return success

def get_today_attendance():
    conn = get_db_connection()
    cursor = conn.cursor()
    today_str = datetime.date.today().isoformat()
    
    # Join with students to get names and metadata
    cursor.execute('''
    SELECT s.student_id, s.name, s.course_section, a.time, a.confidence, a.status
    FROM students s
    LEFT JOIN attendance a ON s.student_id = a.student_id AND a.date = ?
    ORDER BY 
      CASE a.status 
        WHEN 'Present' THEN 1 
        WHEN 'Late' THEN 2 
        WHEN 'Absent' THEN 3 
        ELSE 4 
      END,
      a.time DESC,
      s.name ASC
    ''', (today_str,))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Format results
    logs = []
    for r in rows:
        status = r['status'] if r['status'] is not None else 'Absent'
        logs.append({
            'student_id': r['student_id'],
            'name': r['name'],
            'course_section': r['course_section'],
            'time': r['time'] if r['time'] else '—',
            'confidence': f"{round(r['confidence'], 1)}%" if r['confidence'] else '—',
            'status': status
        })
    return logs

def get_dashboard_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    today_str = datetime.date.today().isoformat()
    
    # 1. Total Students
    cursor.execute('SELECT COUNT(*) FROM students')
    total_students = cursor.fetchone()[0]
    
    # 2. Today stats
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ? AND status IN ('Present', 'Late')", (today_str,))
    present_today = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Absent'", (today_str,))
    absent_today = cursor.fetchone()[0]
    
    # Make sure we account for unrecorded students today as Absent
    if present_today + absent_today < total_students:
        absent_today = total_students - present_today
        
    # 3. Overall Average Attendance rate (percentage)
    # Calculate for each student: (days present/late) / total_days
    cursor.execute('''
    SELECT 
      CAST(SUM(CASE WHEN status IN ('Present', 'Late') THEN 1 ELSE 0 END) AS REAL) / COUNT(*) * 100 as rate
    FROM attendance
    ''')
    avg_row = cursor.fetchone()
    avg_attendance = int(round(avg_row['rate'])) if avg_row and avg_row['rate'] is not None else 83
    
    # 4. Low attendance alerts (below 75%)
    # Group by student, get their rate
    cursor.execute('''
    SELECT s.student_id, s.name, s.course_section,
           CAST(SUM(CASE WHEN a.status IN ('Present', 'Late') THEN 1 ELSE 0 END) AS REAL) / COUNT(a.id) * 100 as rate
    FROM students s
    JOIN attendance a ON s.student_id = a.student_id
    GROUP BY s.student_id
    HAVING rate < 75.0
    ORDER BY rate ASC
    ''')
    low_rows = cursor.fetchall()
    low_attendance_alerts = []
    for r in low_rows:
        low_attendance_alerts.append({
            'student_id': r['student_id'],
            'name': r['name'],
            'course_section': r['course_section'] if r['course_section'] else '—',
            'rate': int(round(r['rate']))
        })
        
    conn.close()
    
    return {
        'total_students': total_students,
        'present_today': present_today,
        'absent_today': absent_today,
        'avg_attendance': avg_attendance,
        'low_attendance_alerts': low_attendance_alerts
    }

def get_report_data(date_str, course_filter):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get daily logs for specific date and course
    query = '''
    SELECT s.student_id, s.name, s.course_section, a.time, a.confidence, a.status,
           (SELECT CAST(SUM(CASE WHEN sub_a.status IN ('Present', 'Late') THEN 1 ELSE 0 END) AS REAL) / COUNT(sub_a.id) * 100 
            FROM attendance sub_a WHERE sub_a.student_id = s.student_id) as total_pct
    FROM students s
    LEFT JOIN attendance a ON s.student_id = a.student_id AND a.date = ?
    WHERE 1=1
    '''
    params = [date_str]
    
    if course_filter and course_filter != 'BCA — All Sections' and course_filter != 'All':
        query += ' AND s.course_section LIKE ?'
        params.append(f"%{course_filter}%")
        
    query += ' ORDER BY s.student_id ASC'
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    report = []
    for r in rows:
        roll = r['student_id'].split('/')[-1] if '/' in r['student_id'] else r['student_id']
        total_pct = int(round(r['total_pct'])) if r['total_pct'] is not None else 83
        
        status = r['status'] if r['status'] is not None else 'Absent'
        report.append({
            'roll': roll,
            'student_id': r['student_id'],
            'name': r['name'],
            'time': r['time'] if r['time'] else '—',
            'confidence': f"{int(round(r['confidence']))}%" if r['confidence'] else '—',
            'total_pct': f"{total_pct}%",
            'status': status
        })
    return report

if __name__ == '__main__':
    # Initialize when run directly
    init_db()
    print("Database initialized successfully at:", DB_FILE)
