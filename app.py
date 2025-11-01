from flask import Flask, render_template, request, redirect, url_for
from fastai.vision.all import load_learner, PILImage
import os
import pathlib
import google.generativeai as genai
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key")

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///animal_game.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    level = db.Column(db.Integer, default=1)
    discovered_animals = db.Column(db.Text, default='')  # comma-separated list
    badge = db.Column(db.String(100), default="🐣 Beginner Explorer")  # 🆕 Add badge column

    def has_discovered(self, animal):
        animals = self.discovered_animals.split(',') if self.discovered_animals else []
        return animal.lower() in [a.strip().lower() for a in animals]

    def add_discovery(self, animal):
        if not self.has_discovered(animal):
            animals = self.discovered_animals.split(',') if self.discovered_animals else []
            animals.append(animal)
            self.discovered_animals = ','.join(animals)
            self.level += 1
            self.update_badge()  # 🆕 Automatically update badge

    def get_discovery_count(self):
        return len(self.discovered_animals.split(',')) if self.discovered_animals else 0

    def update_badge(self):
        count = self.get_discovery_count()
        if count < 5:
            self.badge = "🐣 Beginner Explorer"
        elif count < 10:
            self.badge = "🦊 Forest Adventurer"
        elif count < 20:
            self.badge = "🦁 Wildlife Ranger"
        else:
            self.badge = "🐉 Legendary Zoologist"

model_path = os.path.join(os.path.dirname(__file__), "animal90_classifier.pkl")
model = load_learner(model_path)

def get_animal_fact(animal_name):
    try:
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""
        You are a wildlife educator for kids aged 8–12.
        Give a short, factual and specific description (3–4 sentences) of the animal "{animal_name}".
        Include where it lives, what it eats, and one interesting behavior.
        """
        response = gemini_model.generate_content(prompt)
        text = response.text.strip()
        if not text or len(text.split()) < 5:
            raise ValueError("Response too short or empty")
        return text
    except Exception as e:
        print("Gemini fact generation failed:", e)
        return f"{animal_name.capitalize()}s are fascinating animals found in different parts of the world!"

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('home.html')  

@app.route('/index')
@login_required
def index():
    return render_template('index.html', level=current_user.level)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return "Username already exists! <a href='/register'>Try again</a>"

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            return "Invalid username or password! <a href='/login'>Try again</a>"
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    upload_folder = os.path.join("static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)

    img = PILImage.create(filepath)
    pred, pred_idx, probs = model.predict(img)

    fun_fact = get_animal_fact(pred)

    is_new = False
    if not current_user.has_discovered(pred):
        current_user.add_discovery(pred)
        db.session.commit()
        is_new = True

    image_path = f"uploads/{file.filename}"

    return render_template(
        'result.html',
        image_path=image_path,
        prediction=pred.capitalize(),
        confidence=f"{probs[pred_idx] * 100:.2f}%",
        fun_fact=fun_fact,
        level=current_user.level,
        badge=current_user.badge,
        is_new=is_new
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
