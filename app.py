from flask import Flask, render_template, request , session
from pymongo import MongoClient
from bson import ObjectId
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

client = MongoClient("mongodb://localhost:27017/")
db = client["test-database"]

imageCount = 20


# Face detector model from cascade classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# If these directories don't exist, create them

if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

app = Flask(__name__)

app.secret_key = '2119069_secret_key'

# data base
db = client["attendance_database"]

# collections
login = db["login"]
teacher = db["teachers"]
attendanceDB = db["attendance"]
student = db["students"]

# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(
            gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Adding Attendance to MongoDb database
def add_attendance(name, a_id):
    student_id = name.split('_')[1]
    student_data = student.find_one({'student_id': student_id})
    student_name = student_data['student_name']
    student_rollno = student_data['rollno']
    a_id = ObjectId(a_id)
    if attendanceDB.find_one({'_id': a_id, 'Students': {'$elemMatch': {'$in': [student_id, student_rollno, student_name]}}}):
        print("Attendance Already Marked")
    else:
        attendanceDB.update_one({'_id': a_id}, {'$push': {'Students': [
                                student_id, student_rollno, student_name]}})
        print("Attendance Marked Successfully")


# All routes
@app.route('/')
def home():
    return render_template('index.html', msg="Please Login in or Registered if not done")


@app.route('/insert', methods=['GET', 'POST'])
def insert():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')

    user_data = {
        'name': name,
        'email': email,
        'password': password
    }
    print(user_data)
    login.insert_one(user_data)

# teacher registration
@app.route('/openRegister', methods=['GET', 'POST'])
def openRegister():
    return render_template('register.html', msg="")


@app.route('/openLogin', methods=['GET', 'POST'])
def openLogin():
    return render_template('login.html', msg="")


@app.route('/register', methods=['GET', 'POST'])
def register():
    name = request.form.get('name')
    t_id = request.form.get('t_id')
    password = request.form.get('password')
    t_data = {
        'name': name,
        't_id': t_id,
        'password': password
    }
    print(t_data)
    if(teacher.find_one({'t_id': t_id})):
        return render_template('register.html', msg="Already Registered, Please Login")
    else:
        teacher.insert_one(t_data)
        return render_template('login.html', msg="Registered Successfully, Please Login")


@app.route('/login', methods=['GET', 'POST'])
def login():
    name = request.form.get('name')
    t_id = request.form.get('t_id')
    password = request.form.get('password')
    data = teacher.find_one({'name': name, 't_id': t_id, 'password': password})
    print(data)
    if (data):
        session['t_id'] = True
        return render_template('main.html', name=name, t_id=t_id)
    else:
        print('Not Registered, Please Register first')
        return render_template('login.html', msg="Not Registered, Please Register first")
    
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('t_id', None)
    return render_template('login.html', msg="Logged Out Successfully")

@app.route('/main/<t_id>', methods=['GET', 'POST'])
def main(t_id):
    if 't_id' not in session:
        return render_template('login.html', msg="Please Login First")
    data = teacher.find_one({'t_id': t_id})
    name = data['name']
    return render_template('main.html', name=name, t_id=t_id)


@app.route('/attendance/<name>', methods=['GET', 'POST'])
def attendance(name):
    if 't_id' not in session:
        return render_template('login.html', msg="Please Login First")
    date = request.form.get('date')
    time = request.form.get('time')
    subject = request.form.get('subject')
    course = request.form.get('course')
    semester = request.form.get('semester')
    section = request.form.get('section')
    t_id = teacher.find_one({'name': name})
    attendance_data = {
        'Teacher name': name,
        't_id': t_id,
        'subject': subject,
        'course': course,
        'semester': semester,
        'section': section,
        'date': date,
        'time': time,
        'Students': [[]]
    }
    a_id = attendanceDB.insert_one(attendance_data).inserted_id
    return render_template('attendance.html', msg="Start Attendance", teacher=name, t_id=t_id, a_id=a_id, subject=subject, course=course, semester=semester, section=section)


@app.route('/studentRegister/<t_id>', methods=['GET', 'POST'])
def studentRegister(t_id):
    if 't_id' not in session:
        return render_template('login.html', msg="Please Login First")
    teacherData = teacher.find_one({'t_id': t_id})
    teacherName = teacherData['name']
    return render_template('registerStudent.html',t_id = t_id , name=teacherName, msg="Please Register Student")


@app.route('/registerStudent/<t_id>', methods=['GET', 'POST'])
def registerStudent(t_id):
    if 't_id' not in session:
        return render_template('login.html', msg="Please Login First")
    username = request.form.get('name')
    roll_no = request.form.get('rollno')
    student_id = request.form.get('student_id')
    tname = teacher.find_one({'t_id': t_id})
    student_data = {
        "rollno": roll_no,
        "student_id": student_id,
        "student_name": username
    }
    if student.find_one({'student_id': student_id}):
        return render_template('registerStudent.html',t_id=t_id, name = tname,msg="Student Already Registered")
    student.insert_one(student_data)
    userimagefolder = 'static/faces/'+username+'_'+str(student_id)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images taken for the Dataset: {i}/{imageCount}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Registering User Please Wait...', (30, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = username+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == imageCount*5:
            break
        cv2.imshow('Adding new User....', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    return render_template('registerStudent.html',t_id=t_id, name = tname,msg="Student Registered Successfully")


@app.route('/startAttendance/<a_id>', methods=['GET', 'POST'])
def startAttendance(a_id):
  # cv wala code jisme face detection hoga
    if 't_id' not in session:
        return render_template('login.html', msg="Please Login First")
    print(a_id)
    a_id = ObjectId(a_id)
    currentData = attendanceDB.find_one(a_id)
    tname = currentData['Teacher name']
    subject = currentData['subject']
    course = currentData['course']
    semester = currentData['semester']
    section = currentData['section']
    t_id = currentData['t_id']['t_id']
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('attendance.html', msg="No Students Registered", name=tname, t_id=t_id, a_id=a_id, subject=subject, course=course, semester=semester, section=section)

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame,'Press Esc to Exit', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Attendance', frame) 
            if cv2.waitKey(1) == 27:
                break
    add_attendance(identified_person, a_id)
    cap.release()
    cv2.destroyAllWindows()
    identified_person = identified_person.split('_')[0]
    msg = "Attendance of "+ identified_person + " marked successfully"
    return render_template('attendance.html', msg=msg, name=tname, t_id=t_id, a_id=a_id, subject=subject, course=course, semester=semester, section=section)


@app.route('/viewAttendance/<t_id>', methods=['GET', 'POST'])
def viewAttendance(t_id):
    if 't_id' not in session:
        return render_template('login.html', msg="Please Login First")
    currentTeacher = teacher.find_one({'t_id': t_id})
    teacherName = currentTeacher['name']
    data = [[]]
    return render_template('viewAttendance.html', t_id=t_id, name=teacherName, data=data, msg="View Attendance")


@app.route('/getAttendance/<t_id>', methods=['GET', 'POST'])
def getAttendance(t_id):
    if 't_id' not in session:
        return render_template('login.html', msg="Please Login First")
    date = request.form.get('date')
    time = request.form.get('time')
    subject = request.form.get('subject')
    course = request.form.get('course')
    semester = request.form.get('semester')
    section = request.form.get('section')
    dataAt = attendanceDB.find_one({'t_id.t_id': t_id, 'date': date, 'time': time,
                                   'subject': subject, 'course': course, 'semester': semester, 'section': section})
    data = dataAt['Students']
    print(data)
    currentTeacher = teacher.find_one({'t_id': t_id})
    teacherName = currentTeacher['name']
    l = len(data)
    return render_template('viewAttendance.html', l=l, t_id=t_id, teacherName=teacherName, data=data, msg="View Attendance")


if __name__ == '__main__':
    app.run(debug=True)
