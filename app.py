from flask import Flask, render_template, url_for, request, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import OMR_main
from werkzeug.utils import secure_filename
# from forms import LoginForm

UPLOAD_FOLDER = 'static/uploads/'


app = Flask(__name__)
app.config['SECRET_KEY'] = '53d630d6d8@5dd0fd9a45670bf84168'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 2048 * 2048
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///result.db'
db = SQLAlchemy(app)



ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.')[-1].lower() in ALLOWED_EXTENSIONS



USERNAME1 =  'Admin1'
PASSWORD1 =  'aaa'

USERNAME2 = 'Admin2'
PASSWORD2 = 'OpenIt2000'

questions, choices = 5, 5
solution = [0, 2, 0, 1, 4]
positive = 2
negative = - 0.5

class Scores(db.Model):
    rollno = db.Column(db.Integer , primary_key = True)
    score = db.Column(db.String, nullable = False)
    percentage = db.Column(db.String(4), nullable = False , default = '0%')
    response_sheet = db.Column(db.String(255), nullable = False )
    date_evaluated = db.Column(db.DateTime, nullable = False , default = datetime.utcnow() )

    def __repr__(self):
        return f'{self.rollno}'



#
# @app.route('/login', methods = ['GET','POST'])
# def login():
#     form  = LoginForm()
#     if form.validate_on_submit():
#         flash(f'{form.username.data} Successfully Logged in ', category = "success")
#         return redirect(url_for('upload'))
#     return render_template('index.html', title = 'Login', form = form)
#



@app.route("/", methods = ['POST','GET'])
def load_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if ((username == USERNAME1) and (password == PASSWORD1)) or ((username == USERNAME2) and (password == PASSWORD2)):
            flash("Successfully Logged In ", category = 'success')
            return redirect('upload')
        else:
            flash("Invalid Username or Password", category = 'error')
            return redirect(url_for('load_login'))
    else:
        print('get request')
        return render_template('index.html')



@app.route("/score", methods = ['POST'])
def upload():
    if 'file' not in request.files:
        flash("Please fill the data required", category='error')
        return redirect('/upload')
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading',category = 'error')
        return redirect('/upload')
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            evaluated_image = OMR_main.evaluate_result(f'static/uploads/{filename}',solution,positive,negative)
            print("....**********************...", request.url)

            # flash('Image successfully uploaded and displayed below',category='success')
            return render_template('score.html', filename=filename, evaluated_image = evaluated_image, marking_scheme = [positive,negative])
        except:
            flash('Please Upload Correct Photo of OMR Sheet', category='success')
            return redirect(url_for('upload_form'))
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif', category='success')
        return redirect('/upload')

@app.route("/upload")
def upload_form():
    return render_template('upload.html')


@app.route('/upload/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)
#
# def get_evaluated_image(filename):
#     return OMR_main.evaluate_result(filename)


# to add new student's result to scoreboard
@app.route('/scoreboard', methods  = ['POST'])
def add_score():
    print("..................",request.url)

    if request.method=='POST':
        all_scores = Scores.query.order_by(Scores.rollno).all()
        try:
            rollno = int(request.form['rollno'])
            score = request.form['score']
            percentage = request.form['percentage']
            response_sheet = request.form['image_name']
            new_score = Scores(rollno = rollno, score = score, percentage = percentage, response_sheet = response_sheet)
        except:
            flash("Please upload correct photo", category='error')
            return redirect(url_for('upload_form'))

        try:
            db.session.add(new_score)
            db.session.commit()
            flash("Student score successfully added", category='success')
            return render_template('scoreboard.html', scores=all_scores)
        except:
            flash("This record is already present", category='success')
            return redirect(url_for('upload_form'))




@app.route("/scoreboard")
def show_scoreboard():
    all_scores = Scores.query.order_by(Scores.rollno).all()
    return render_template('scoreboard.html', scores=all_scores)





@app.route('/delete/<int:rollno>')
def delete(rollno):
    score = Scores.query.get_or_404(rollno)
    db.session.delete(score)
    db.session.commit()
    return redirect('/scoreboard')


@app.route('/edit/<int:rollno>',methods = ['POST','GET'])
def edit(rollno):
    score = Scores.query.get_or_404(rollno)
    if request.method == 'POST':
        score.rollno = request.form['rollno']
        score.score = request.form['score']
        score.percentage = request.form['percentage']
        db.session.commit()
        return redirect('/scoreboard')
    else:
        return render_template('edit.html', score = score)


@app.route('/marking_scheme',methods = ['POST','GET'])
def marking_scheme():
    if request.method == 'POST':
        global positive
        global negative
        positive = float(request.form['correct'])
        negative = float(request.form['wrong'])
        flash("Marking Scheme successfully updated !", category='success')
        return redirect(url_for('marking_scheme'))
    else:
        return render_template('marking_scheme.html',pos = positive, neg = negative)


@app.route('/answer_key',methods = ['POST','GET'])
def answer_key():
    if request.method == 'POST':
        global solution
        solution = []
        for i in range(5):
            option = request.form[str(i+1)]
            solution.append(int(option))
        flash("Answer Key successfully updated !", category='success')
        return redirect(url_for('answer_key'))
    else:
        return render_template('answer_key.html')





#
# @app.route("/scoreboard")
# def show_scoreboard():
#     return render_template('scoreboard.html')

# @app.route("/edit")
# def edit():
#     return render_template('edit.html')

@app.route("/activity")
def show_activity():
    return render_template('activity.html')






























if __name__ == '__main__':
        app.run(debug = True)