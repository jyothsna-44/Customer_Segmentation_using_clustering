import os
import pickle
import pandas as pd
import plotly.express as px
import plotly.io as pio
from flask import Flask, render_template, redirect, url_for, request, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# --- CONFIGURATION ---
app.config['SECRET_KEY'] = 'mall_ai_ultra_pro_2024_secure'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///customer_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- ML MODEL LOADING ---
try:
    model = pickle.load(open('kmeans_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    print(f"Model Error: {e}")

segment_info = [
    {'name': 'Sensible', 'strategy': 'Focus on budget-friendly discount alerts.', 'icon': 'fa-piggy-bank', 'color': '#00d2ff'},
    {'name': 'Standard Group', 'strategy': 'Maintain engagement via newsletters.', 'icon': 'fa-user-friends', 'color': '#3b82f6'},
    {'name': 'Target Customers', 'strategy': 'High priority! Offer VIP rewards and luxury previews.', 'icon': 'fa-crown', 'color': '#8b5cf6'},
    {'name': 'Careful Spenders', 'strategy': 'Encourage spending with high-value cashback.', 'icon': 'fa-shield-halved', 'color': '#10b981'},
    {'name': 'Spendthrifts', 'strategy': 'Target with frequent flash sales.', 'icon': 'fa-bolt-lightning', 'color': '#f59e0b'}
]

# --- ROUTES ---

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    df = pd.read_excel('Mall_Customers.xlsx')
    X_scaled = scaler.transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    df['Cluster'] = model.predict(X_scaled)
    df['Segment'] = df['Cluster'].apply(lambda x: segment_info[x]['name'])

    fig = px.scatter_3d(df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                        color='Segment', template="plotly_dark", height=500,
                        color_discrete_map={s['name']: s['color'] for s in segment_info})
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), paper_bgcolor='rgba(0,0,0,0)')
    graph_html = pio.to_html(fig, full_html=False, config={'displayModeBar': True})

    if 'history' not in session: session['history'] = []
    prediction = None
    l_inc, l_spn = "", ""

    if request.method == 'POST':
        if 'clear_history' in request.form:
            session['history'] = []
            session.modified = True
        else:
            l_inc = request.form.get('income')
            l_spn = request.form.get('spending')
            res_id = model.predict(scaler.transform([[float(l_inc), float(l_spn)]]))[0]
            prediction = segment_info[res_id]
            session['history'].insert(0, {'inc': l_inc, 'spn': l_spn, 'name': prediction['name'], 'color': prediction['color']})
            session.modified = True

    stats = {'total': len(df), 'avg_inc': round(df['Annual Income (k$)'].mean(), 1), 'top': df['Segment'].value_counts().idxmax()}
    return render_template('dashboard.html', name=current_user.username, graph=graph_html, prediction=prediction, 
                           last_inc=l_inc, last_spn=l_spn, history=session['history'], insights=segment_info, stats=stats)

@app.route('/customers')
@login_required
def customers():
    df = pd.read_excel('Mall_Customers.xlsx')
    X_scaled = scaler.transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    df['Cluster'] = model.predict(X_scaled)
    data = []
    for _, row in df.iterrows():
        seg = segment_info[row['Cluster']]
        data.append({'id': row['CustomerID'], 'age': row['Age'], 'income': row['Annual Income (k$)'], 
                     'score': row['Spending Score (1-100)'], 'segment': seg['name'], 'color': seg['color']})
    return render_template('customers.html', data=data)

@app.route('/reports')
@login_required
def reports():
    df = pd.read_excel('Mall_Customers.xlsx')
    X_scaled = scaler.transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    df['Cluster'] = model.predict(X_scaled)
    counts = df['Cluster'].value_counts().sort_index()
    
    labels = [s['name'] for s in segment_info]
    values = [int(counts.get(i, 0)) for i in range(len(segment_info))]
    colors = [s['color'] for s in segment_info]

    stats = {'total': len(df), 'avg_inc': round(df['Annual Income (k$)'].mean(), 2), 
             'avg_spn': round(df['Spending Score (1-100)'].mean(), 2), 
             'max_inc': df['Annual Income (k$)'].max(), 'min_inc': df['Annual Income (k$)'].min()}
    return render_template('reports.html', stats=stats, labels=labels, values=values, colors=colors)

@app.route('/logout')
def logout():
    logout_user(); return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context(): db.create_all()
    app.run(debug=True)