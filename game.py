# app1.py
import os
from flask import Flask, render_template, request, jsonify, session, url_for, redirect, current_app, g, make_response, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime, timedelta
import uuid
from uuid import uuid4

import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
from telebot import apihelper
from telebot import TeleBot, types
import hashlib
import hmac
from urllib.parse import parse_qsl, urlparse, urlencode, parse_qs
from threading import Thread
import requests
import logging
#from pyngrok import ngrok, conf
import json
import time
from sqlalchemy import BigInteger, func, cast, String, select, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import SQLAlchemyError
from flask import session as flask_session
from flask.cli import AppGroup
from functools import wraps
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine
from flask_jwt_extended import JWTManager, jwt_required, verify_jwt_in_request
from flask_jwt_extended import (
    get_jwt_identity,
    create_access_token, create_refresh_token
)
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from werkzeug.exceptions import BadRequest, NotFound, Unauthorized, InternalServerError
from sqlalchemy.orm.exc import NoResultFound
from werkzeug.middleware.proxy_fix import ProxyFix
# Load environment variables
load_dotenv()

# Konfiguracja podstawowa logowania
logging.basicConfig(level=logging.INFO)

# Utworzenie obiektu loggera
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['GAME_URL'] = os.getenv('GAME_URL')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
GAME_URL = app.config['GAME_URL']


# Configure session to use filesystem
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

# Initialize SQLAlchemy
db = SQLAlchemy(app)
jwt = JWTManager(app)
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
Session = scoped_session(sessionmaker(bind=engine))

# Initialize Telegram Bot
bot = telebot.TeleBot(os.getenv('TELEGRAM_BOT_TOKEN'))
bot_cli = AppGroup('bot')

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Konfiguracja cachowania
cache_config = {
    '.js': {'max-age': 86400},        # 1 dzień
    '.css': {'max-age': 86400},       # 1 dzień
    '.jpg': {'max-age': 604800},      # 1 tydzień
    '.png': {'max-age': 604800},      # 1 tydzień
    'default': {'max-age': 300}       # 5 minut
}

@app.route('/static/<path:filename>')
def serve_static(filename):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(root_dir, 'static')

    # Sprawdź, czy plik istnieje
    if not os.path.exists(os.path.join(static_dir, filename)):
        return "File not found", 404

    # Ustal typ pliku i odpowiednie ustawienia cache
    _, ext = os.path.splitext(filename)
    cache_settings = cache_config.get(ext, cache_config['default'])

    response = send_from_directory(static_dir, filename)
    response.cache_control.public = True
    response.cache_control.max_age = cache_settings['max-age']

    # Dodaj nagłówek ETag dla efektywnej walidacji cache
    response.add_etag()

    return response.make_conditional(request)


# Models
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    telegram_id = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=False, nullable=False)
    wallet_address = db.Column(db.String(120), unique=True)

class UserProgress(db.Model):
    __tablename__ = 'user_progress'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    level = db.Column(db.Integer, default=1)
    experience = db.Column(db.Integer, default=0)
    coins = db.Column(db.Float, default=0)
    energia = db.Column(db.Integer, default=100)
    max_energia = db.Column(db.Integer, default=100)
    gems = db.Column(db.Float, default=0)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserBuilding(db.Model):
    __tablename__ = 'UserBuildings'
    user_building_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    building_id = db.Column(db.Integer, db.ForeignKey('Buildings.building_id'), nullable=False)
    quantity = db.Column(db.Integer, default=0)

class Building(db.Model):
    __tablename__ = 'Buildings'
    building_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(255))
    cost = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Boost(db.Model):
    __tablename__ = 'Boosts'
    boost_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(255))
    effect = db.Column(db.Float, nullable=False)
    duration = db.Column(db.Integer, nullable=False)
    cost = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserBoost(db.Model):
    __tablename__ = 'UserBoosts'
    user_boost_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.telegram_id'), nullable=False)
    boost_id = db.Column(db.Integer, db.ForeignKey('Boosts.boost_id'), nullable=False)
    quantity = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ReferralLink(db.Model):
    __tablename__ = 'Reflinks'
    reflink_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    reflink_code = db.Column(db.String(20), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserShop(db.Model):
    __tablename__ = 'UserShop'
    purchase_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    item_id = db.Column(db.Integer, nullable=False)
    item_level = db.Column(db.Integer, nullable=False)
    coins = db.Column(db.Float, nullable=False)
    gems = db.Column(db.Float, nullable=False)

class UserWallet(db.Model):
    __tablename__ = 'user_wallet'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    address = db.Column(db.String(120), unique=True, nullable=False)

class AppSession(db.Model):
    __tablename__ = 'session'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.telegram_id'), nullable=False)  # Correctly reference User.id
    init_data = db.Column(db.Text, nullable=False)
    start_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('session', lazy=True))

# Error Handlers
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal Server Error: {str(error)}", exc_info=True)
    return jsonify({"error": "Internal Server Error", "details": str(error)}), 500

def verify_token(token, secret_key, algorithms=['HS256']):
    """Dekoduje i weryfikuje token JWT."""
    try:
        payload = jwt.decode(token, secret_key, algorithms=algorithms)
        return payload
    except ExpiredSignatureError:
        logger.warning("Expired token attempted")
        return None
    except InvalidTokenError:
        logger.warning("Invalid token attempted")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in token verification: {str(e)}")
        return None

# Funkcje pomocnicze
def get_user_by_telegram_id(telegram_id):
    user = User.query.filter_by(telegram_id=telegram_id).first()
    return user

def get_or_create_user(telegram_id, username, email=None):
    user = get_user_by_telegram_id(telegram_id)
    if not user:
        user = User(
            telegram_id=telegram_id,
            username=username or f"Użytkownik_{telegram_id}",
            email=email or f"{username}@yetiai.pl"
        )
        db.session.add(user)
        db.session.commit()
    return user

def create_initial_user_progress(user_id):
    try:
        # Sprawdź, czy użytkownik już ma progress
        existing_progress = UserProgress.query.filter_by(user_id=int(user_id)).first()
        if existing_progress:
            return existing_progress

        # Jeśli nie ma, stwórz nowy progress
        new_progress = UserProgress(
            user_id=int(user_id),
            level=1,
            experience=0,
            coins=0,
            energia=100,
            updated_at=datetime.utcnow()
        )
        db.session.add(new_progress)

        # Dodaj początkowe budynki (jeśli są)
        initial_buildings = Building.query.filter(Building.cost == 0).all()
        for building in initial_buildings:
            user_building = UserBuilding(
                user_id=user_id,
                building_id=building.building_id,
                quantity=1  # Dajemy po jednym darmowym budynku
            )
            db.session.add(user_building)

        # Dodaj początkowe boosty (jeśli są)
        initial_boosts = Boost.query.filter(Boost.cost == 0).all()
        for boost in initial_boosts:
            user_boost = UserBoost(
                user_id=user_id,
                boost_id=boost.boost_id,
                quantity=1  # Dajemy po jednym darmowym booście
            )
            db.session.add(user_boost)

        db.session.commit()
        return new_progress
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in create_initial_user_progress: {str(e)}", exc_info=True)
        return None

@app.route('/start', methods=['POST'])
def start_game():
    try:
        # Inicjalizacja session_id
        session_id = session.get('session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id

        # Sprawdzenie, czy Content-Type jest 'application/json'
        if request.content_type != 'application/json':
            return jsonify({"error": "Unsupported Media Type"}), 415

        # Odczytaj dane JSON z ciała żądania
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        # Pobierz telegram_id i username
        telegram_id = data.get('telegram_id')
        username = data.get('username')

        if not telegram_id or not username:
            return jsonify({"error": "Brakujące dane: telegram_id lub username"}), 400

        with Session() as db_session:
            # Utwórz lub pobierz użytkownika
            user, is_new_user = get_or_create_user(telegram_id, username)

            # Pobierz lub utwórz progres użytkownika
            progress = db_session.query(UserProgress).filter_by(user_id=user.telegram_id).first()
            if not progress:
                progress = create_initial_user_progress(user.telegram_id)
                db_session.add(progress)
                db_session.commit()

            # Utwórz tokeny dostępu
            access_token = create_access_token(identity=user.telegram_id)
            refresh_token = create_refresh_token(identity=user.telegram_id)

            # Przygotuj dane gry
            game_data = {
                'user_id': user.id,
                'telegram_id': user.telegram_id,
                'username': user.username,
                'session_id': session_id,
                'level': progress.level,
                'experience': progress.experience,
                'coins': progress.coins,
                'energia': progress.energia,
                'access_token': access_token,
                'refresh_token': refresh_token
            }

            return jsonify({'session_id': session_id, 'game_data': game_data})

    except Exception as e:
        logger.error(f"Error in start_game: {str(e)}")
        return jsonify({"error": "Błąd podczas rozpoczynania gry"}), 500


# Modify the set_webhook function:
def set_webhook(WEBHOOK_URL):
    try:
        parsed_url = urlparse(WEBHOOK_URL)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")

        bot.remove_webhook()
        bot.set_webhook(url=WEBHOOK_URL)
        logger.info(f"Webhook set to {WEBHOOK_URL}")
    except ValueError as ve:
        logger.error(f"Invalid webhook URL: {str(ve)}")
        raise  # Rethrow the exception to handle it at a higher level
    except Exception as e:
        logger.error(f"Error setting webhook: {str(e)}")
        raise  # Rethrow the exception for consistent error handling

@app.route('/')
def index():
    login_info = {
        "message": "Musisz się zalogować, aby uzyskać dostęp do tej strony.",
        "login_url": "login"  # Replace with the correct login URL
    }
    return jsonify(login_info)

@app.route('/login', methods=['POST'])
def start_chat():
    chat_session = {'session_id': str(uuid.uuid4())}
    return jsonify(chat_session)

def get_user_and_session(session_id):
    try:
        with Session() as session:
            app_session = session.query(AppSession).filter_by(session_id=session_id).first()
            if app_session:
                user = session.query(User).filter_by(telegram_id=app_session.user_id).first()
                return user, app_session
            else:
                return None, None
    except Exception as e:
        logger.error(f"Error in get_user_and_session: {str(e)}")
        return None, None

@app.route('/game', methods=['GET','POST'])

def game():
    session_id = request.args.get('session_id')
    token = request.args.get('token')
    if not session_id:
        logger.warning("Attempt to access game without session ID")
        return render_template('error.html', error="Nieprawidłowa sesja"), 400

    with Session() as db_session:
        try:
            # Get the session
            session = db_session.query(AppSession).filter_by(session_id=session_id).first()
            if not session:
                logger.warning(f"No session found for session ID: {session_id}")
                return render_template('error.html', error="Sesja nie znaleziona"), 404

            # Get the user using the user_id
            user = db_session.query(User).filter_by(telegram_id=session.user_id).first()
            if not user:
                logger.warning(f"User not found for user_id: {session.user_id}")
                return render_template('error.html', error="Użytkownik nie znaleziony"), 404

            # Fetch necessary game data
            progress = db_session.query(UserProgress).filter_by(user_id=user.telegram_id).first()
            if not progress:
                logger.warning(f"Progress not found for telegram_id: {user.telegram_id}")
                # Create initial progress if it doesn't exist
                progress = create_initial_user_progress(user.telegram_id)
                if not progress:
                    return render_template('error.html', error="Nie można utworzyć postępu użytkownika"), 500

            # Ensure all necessary fields are present and have default values if None
            game_data = {
                'user_id': user.telegram_id,
                'telegram_id': user.telegram_id,
                'username': user.username,
                'session_id': session_id,
                'token': token,
                'level': progress.level or 1,
                'experience': progress.experience or 0,
                'coins': progress.coins or 0,
                'energia': progress.energia or 0,
                'max_energia': progress.max_energia or 100
            }

            logger.info(f"Game data prepared for user {user.telegram_id}: {game_data}")

            return render_template('game.html', game_data=game_data)

        except Exception as e:
            logger.error(f"Error in game route: {str(e)}", exc_info=True)
            return render_template('error.html', error="Wystąpił nieoczekiwany błąd"), 500

def get_session_by_id(session_id):
    return AppSession.query.filter_by(session_id=session_id).first()

def get_user_by_id(user_id):
    return User.query.filter_by(telegram_id=user_id).first()

@app.route('/api/user/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    current_user = get_jwt_identity()
    new_access_token = create_access_token(identity=current_user)
    return jsonify(access_token=new_access_token), 200

@app.errorhandler(ExpiredSignatureError)
def handle_expired_token(error):
    return jsonify({"error": "Token wygasł. Proszę odświeżyć."}), 401

@app.errorhandler(InvalidTokenError)
def handle_invalid_token(error):
    return jsonify({"error": "Nieprawidłowy token."}), 401

@app.route('/api/user/stats', methods=['POST'])
@jwt_required()
def get_user_stats():
    try:
        current_user = get_jwt_identity()  # Pobierz identyfikator zalogowanego użytkownika z tokena JWT
        user = User.query.filter_by(telegram_id=current_user).first()

        if not user:
            return jsonify({"error": "Nie znaleziono użytkownika"}), 404

        progress = UserProgress.query.filter_by(user_id=user.telegram_id).first()
        if not progress:
            progress = create_initial_user_progress(user.telegram_id)

        return jsonify({
            'username': user.username,
            'level': progress.level,
            'experience': progress.experience,
            'coins': progress.coins,
            'energia': progress.energia,
            'max_energia': progress.max_energia,
            'gems': progress.gems
        })
    except Exception as e:
        logger.error(f"Error in get_user_stats: {str(e)}")
        return jsonify({"error": "Błąd podczas pobierania statystyk użytkownika"}), 500

@app.route('/api/buildings/buy', methods=['POST'])
@jwt_required()
def buy_building():
    data = request.json
    building_id = data.get('building_id')
    current_user = get_jwt_identity()

    try:
        with Session() as session:
            building = session.query(Building).get(building_id)
            user = session.query(User).filter_by(telegram_id=current_user).first()
            user_progress = session.query(UserProgress).filter_by(user_id=user.telegram_id).first()
            user_building = session.query(UserBuilding).filter_by(user_id=user.telegram_id, building_id=building_id).first()

            if not user_building:
                user_building = UserBuilding(user_id=user.telegram_id, building_id=building_id, quantity=0)
                session.add(user_building)

            if user_progress.coins < building.cost:
                return jsonify({'error': 'Not enough coins'}), 400

            user_progress.coins -= building.cost
            user_building.quantity += 1

            session.commit()

            return jsonify({'success': True, 'building_id': building_id, 'new_quantity': user_building.quantity})
    except Exception as e:
        logger.error(f"Error in buy_building: {str(e)}")
        return jsonify({'error': 'Failed to buy building'}), 500

@app.route('/api/lottery/play', methods=['POST'])
@jwt_required()
def play_lottery():
    try:
        current_user = get_jwt_identity()  # Pobierz identyfikator zalogowanego użytkownika z tokena JWT

        with Session() as session:
            user = session.query(User).filter_by(telegram_id=current_user).first()
            if not user:
                return jsonify({"error": "Nie znaleziono użytkownika"}), 404

            progress = session.query(UserProgress).filter_by(user_id=user.telegram_id).first()
            if not progress:
                return jsonify({'error': 'User progress not found'}), 404

            prize = 100 # przykładowa wygrana
            progress.coins += prize
            progress.energia -= 50
            session.commit()
            return jsonify({'prize': prize}), 200
    except Exception as e:
        logger.error(f"Error in /api/lottery/play: {e}")
        return jsonify({'error': 'Failed to play lottery'}), 500


@app.route('/api/user/update', methods=['POST'])
@jwt_required()
def update_user_progress():
    """Aktualizuje postęp użytkownika."""
    try:
        current_user = get_jwt_identity()
        data = request.get_json()

        with Session() as db_session:
            progress = db_session.query(UserProgress).filter_by(user_id=current_user).first()

            if progress:
                # Aktualizuj pola tylko jeśli są obecne w przesłanych danych
                for field in ['coins', 'experience', 'level', 'energia', 'max_energia', 'gems']:
                    if field in data:
                        setattr(progress, field, data[field])

                progress.updated_at = datetime.utcnow()
            else:
                # Jeśli nie ma postępu, utwórz nowy
                progress = UserProgress(
                    user_id=current_user,
                    coins=data.get('coins', 0),
                    experience=data.get('experience', 0),
                    level=data.get('level', 1),
                    energia=data.get('energia', 100),
                    max_energia=data.get('max_energia', 100),
                    gems=data.get('gems', 0)
                )
                db_session.add(progress)

            db_session.commit()

            return jsonify({
                'message': 'Postęp użytkownika został zaktualizowany',
                'updated_progress': {
                    'coins': progress.coins,
                    'experience': progress.experience,
                    'level': progress.level,
                    'energia': progress.energia,
                    'max_energia': progress.max_energia,
                    'gems': progress.gems
                }
            }), 200

    except SQLAlchemyError as e:
        db_session.rollback()
        logger.error(f"Database error updating user progress: {str(e)}", exc_info=True)
        return jsonify({'error': 'Błąd bazy danych podczas aktualizacji postępu użytkownika'}), 500
    except Exception as e:
        logger.error(f"Unexpected error updating user progress: {str(e)}", exc_info=True)
        return jsonify({'error': 'Nieoczekiwany błąd podczas aktualizacji postępu użytkownika'}), 500

def create_initial_user_progress(user_id):
    try:
        # Sprawdź, czy użytkownik już ma progress
        existing_progress = UserProgress.query.filter_by(user_id=int(user_id)).first()
        if existing_progress:
            return existing_progress

        # Jeśli nie ma, stwórz nowy progress
        new_progress = UserProgress(
            user_id=int(user_id),
            level=1,
            experience=0,
            coins=0,
            energia=100,
            updated_at=datetime.utcnow()
        )
        db.session.add(new_progress)

        # Dodaj początkowe budynki (jeśli są)
        initial_buildings = Building.query.filter(Building.cost == 0).all()
        user_buildings = [
            UserBuilding(user_id=user_id, building_id=building.building_id, quantity=1)
            for building in initial_buildings
        ]
        db.session.bulk_save_objects(user_buildings)

        # Dodaj początkowe boosty (jeśli są)
        initial_boosts = Boost.query.filter(Boost.cost == 0).all()
        user_boosts = [
            UserBoost(user_id=user_id, boost_id=boost.boost_id, quantity=1)
            for boost in initial_boosts
        ]
        db.session.bulk_save_objects(user_boosts)

        db.session.commit()
        return new_progress
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in create_initial_user_progress: {str(e)}", exc_info=True)
        return None

def update_session_activity(session_data):
    session_data.last_activity = datetime.utcnow()
    db.session.commit()

@app.route('/api/leaderboard', methods=['GET'])
@jwt_required()
def get_leaderboard():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))

    # Użyj funkcji rankingowej bazy danych do obliczenia pozycji w rankingu
    top_players = db.session.query(
        UserProgress.user_id,  # Assuming user_id is the foreign key to your User table
        User.username,  # Assuming you have a User table with username
        UserProgress.level,
        UserProgress.coins,
        func.rank().over(order_by=(UserProgress.level.desc(), UserProgress.coins.desc())).label('rank')
    ).join(User, User.id == UserProgress.user_id) \
    .order_by('rank') \
    .paginate(page=page, per_page=per_page, error_out=False)  # Implement pagination

    leaderboard_data = [
        {
            "position": player.rank,
            "username": player.username,
            "level": player.level,
            "coins": player.coins
        }
        for player in top_players.items
    ]
    return jsonify({'leaderboard': leaderboard_data, 'has_next': top_players.has_next})

def calculate_passive_income():
    with app.app_context():
        while True:
            try:
                sql = text("""
                    UPDATE user_progress up
                    JOIN (
                        SELECT BINARY(ub.user_id) as user_id, SUM(ub.quantity * b.cost * 0.1) AS passive_income
                        FROM UserBuildings ub
                        JOIN Buildings b ON ub.building_id = b.building_id
                        GROUP BY ub.user_id
                    ) AS income_subq ON BINARY(up.user_id) = income_subq.user_id
                    SET up.coins = up.coins + CAST(income_subq.passive_income AS DECIMAL(10,2)),
                        up.updated_at = CURRENT_TIMESTAMP
                """)

                db.session.execute(sql)
                db.session.commit()

                logger.info("Obliczanie pasywnego dochodu zakończone pomyślnie")
                time.sleep(60)  # Uruchamiaj co 60 sekund
            except Exception as e:
                logger.error(f"Błąd w obliczaniu pasywnego dochodu: {str(e)}")
                db.session.rollback()

from flask_wtf.csrf import generate_csrf

@app.route('/webhook', methods=['POST'])
def webhook():
    csrf_token = generate_csrf()
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        logger.debug(f"Received update: {update}")
        try:
            bot.process_new_updates([update])
        except Exception as e:
            logger.error(f"Error processing update: {str(e)}")
        return ''
    else:
        logger.warning("Received non-JSON request to webhook")
        return jsonify({'error': 'Invalid content type'}), 403

# This endpoint should be used to delete the webhook
@app.route('/delete-webhook', methods=['GET'])
def delete_webhook():
    try:
        bot.remove_webhook()
        return jsonify({'message': 'Webhook deleted successfully'}), 200
    except Exception as e:
        logger.error(f"Error deleting webhook: {str(e)}")
        return jsonify({'error': 'Failed to delete webhook'}), 500

# Endpoint to start polling
@app.route('/start-polling', methods=['GET'])
def start_polling():
    try:
        bot.remove_webhook()  # Ensure webhook is deleted before starting polling
        bot.polling()
        return jsonify({'message': 'Polling started successfully'}), 200
    except Exception as e:
        logger.error(f"Error starting polling: {str(e)}")
        return jsonify({'error': 'Failed to start polling'}), 500

@bot_cli.command('run')
def run_bot():
    bot.remove_webhook()
    bot.polling()

def handle_play_button(call, session_id):
    try:
        with app.app_context():
            # Pobieranie użytkownika na podstawie Telegram ID
            user, is_new_user = get_user_by_telegram_id(call.from_user.id)

            if not user:
                username = call.from_user.username or f"User_{call.from_user.id}"
                email = f"{username}@yetiai.pl"
                user, is_new_user = get_or_create_user(call.from_user.id, username, email)

            if user:
                # Sprawdzenie i utworzenie progresu użytkownika, jeśli nie istnieje
                progress = UserProgress.query.filter_by(user_id=user.telegram_id).first()
                if not progress:
                    progress = create_initial_user_progress(user.telegram_id)

                # Aktualizacja sesji w bazie danych
                session_data = AppSession.query.filter_by(session_id=session_id).first()
                if session_data:
                    session_data.last_activity = datetime.utcnow()
                    if session_data.start_time is None:
                        session_data.start_time = datetime.utcnow()
                    db.session.commit()

                    app.logger.info(f'Telegram ID: {user.telegram_id} for session {session_id}')

                    # Pobieranie tokena z danych sesji
                    init_data = json.loads(session_data.init_data)
                    token = init_data.get('token')

                    # Tworzenie URL do Telegram WebApp
                    webapp_url = f"{app.config['GAME_URL']}/game?session_id={session_id}&token={token}"

                    # Tworzenie przycisku WebApp
                    web_app_button = InlineKeyboardButton(
                        text="Graj!",
                        web_app=WebAppInfo(url=webapp_url)
                    )

                    # Tworzenie klawiatury z przyciskiem WebApp
                    keyboard = InlineKeyboardMarkup().add(web_app_button)

                    # Wysyłanie wiadomości z przyciskiem WebApp
                    bot.send_message(
                        call.message.chat.id,
                        f"Kliknij przycisk poniżej, aby rozpocząć grę!",
                        reply_markup=keyboard
                    )

                    return {
                        'status': 'success',
                        'session_id': session_id,
                        'token': token,
                        'redirect_url': webapp_url
                    }
                else:
                    app.logger.warning(f'Brak danych sesji dla session_id: {session_id}')
                    bot.answer_callback_query(call.id, "Sesja nie istnieje. Spróbuj ponownie.")
                    return {'status': 'error', 'message': 'Sesja nie istnieje'}

            else:
                app.logger.warning(f'Autoryzacja nie powiodła się dla użytkownika: {call.from_user.id}')
                bot.answer_callback_query(call.id, "Autoryzacja nie powiodła się. Spróbuj ponownie.")
                return {'status': 'error', 'message': 'Autoryzacja nie powiodła się'}

    except Exception as e:
        app.logger.error(f"Błąd w handle_play_button: {str(e)}", exc_info=True)
        bot.answer_callback_query(call.id, "Wystąpił błąd. Spróbuj ponownie później.")
        return {'status': 'error', 'message': 'Wystąpił błąd'}

def get_user_by_telegram_id(telegram_id):
    user = User.query.filter_by(telegram_id=telegram_id).first()
    if user:
        return user, False  # Assuming user is not new
    else:
        return None, True  # New user

def get_or_create_user(telegram_id, username, email=None):
    # Sprawdź, czy istnieje użytkownik o podanym telegram_id
    user = User.query.filter_by(telegram_id=telegram_id).first()
    is_new_user = False

    if not user:
        # Użytkownik nie istnieje, więc tworzymy nowego
        user = User(
            telegram_id=telegram_id,
            username=username or f"Użytkownik_{telegram_id}",
            email=email or f"{username}@yetiai.pl"
        )
        db.session.add(user)
        db.session.commit()
        is_new_user = True

    return user, is_new_user

@bot.message_handler(commands=['start'])
def send_welcome(message):
    try:
        logger.info(f"Received /start command from user {message.chat.id}")

        # Initialize data
        init_data = prepare_init_data(message)
        user_id = message.from_user.id

        with app.app_context():
            # Initialize session if not exists
            if not hasattr(g, 'session_id'):
                initialize_session(message, init_data)

            # Use g.session_id for session management
            if g.session_id:
                markup = prepare_markup(g.session_id)
                welcome_message = (
                    "🎉 Witaj w YetiCoinTap 🎉\n"
                    "Naciśnij przycisk poniżej, aby rozpocząć grę i zdobywać YetiCoiny 🐾"
                )
                sent_message = bot.send_message(message.chat.id, welcome_message, reply_markup=markup)
                logger.info(f"Welcome message sent to user {message.chat.id}: Message ID {sent_message.message_id}")
            else:
                logger.error(f"Failed to create or get session for user {message.chat.id}")
                bot.send_message(message.chat.id, "Przepraszamy, wystąpił błąd podczas rozpoczynania gry. Spróbuj ponownie później.")

    except Exception as e:
        logger.error(f"Error in send_welcome: {str(e)}", exc_info=True)
        bot.send_message(message.chat.id, "Przepraszamy, wystąpił nieoczekiwany błąd. Spróbuj ponownie później.")

def initialize_session(message, init_data):
    try:
        session_id, token = create_or_update_session(message.chat.id, init_data)
        g.session_id = session_id
        g.telegram_id = message.chat.id
        g.username = message.from_user.username
    except Exception as e:
        logger.error(f"Błąd podczas inicjalizacji sesji: {str(e)}")
        # Make sure to set g.session_id to None in case of an error
        g.session_id = None

def create_or_update_session(telegram_id, init_data):
    try:
        current_time = datetime.utcnow()
        session_expiration = timedelta(hours=2)

        with Session() as session:
            # Find the user in the database, if not exists - create
            user = session.query(User).filter_by(telegram_id=telegram_id).first()
            if not user:
                username = init_data.get('username', f"Użytkownik_{telegram_id}")
                email = init_data.get('email', f"{username}@yetiai.pl")
                user = User(telegram_id=telegram_id, username=username, email=email)
                session.add(user)
                session.flush()

            # Check for existing active session
            existing_session = session.query(AppSession).filter(
                AppSession.user_id == str(user.telegram_id),
                AppSession.start_time > (current_time - session_expiration)
            ).order_by(AppSession.start_time.desc()).first()

            if existing_session:
                # Update existing session
                existing_session.last_activity = current_time
                session_id = existing_session.session_id
                logger.info(f"Zaktualizowano istniejącą sesję {session_id} dla użytkownika {user.telegram_id}")
            else:
                # Create new session
                session_id = str(uuid.uuid4())
                new_session = AppSession(
                    session_id=session_id,
                    user_id=str(user.telegram_id),
                    init_data=json.dumps(init_data),
                    start_time=current_time,
                    last_activity=current_time
                )
                session.add(new_session)
                logger.info(f"Utworzono nową sesję {session_id} dla użytkownika {user.telegram_id}")

            # Generate token
            token = create_access_token(identity=telegram_id)

            # Update init_data with session_id and token
            init_data['session_id'] = session_id
            init_data['token'] = token

            # Update the session's init_data
            app_session = session.query(AppSession).filter_by(session_id=session_id).first()
            app_session.init_data = json.dumps(init_data)

            session.commit()
            logger.debug(f"Sesja została pomyślnie zapisana dla użytkownika {user.telegram_id}")
            return session_id, token

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Błąd bazy danych w create_or_update_session: {str(e)}")
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Nieoczekiwany błąd w create_or_update_session: {str(e)}")
        raise

def cleanup_expired_sessions():
    try:
        with Session() as session:
            expiration_time = datetime.utcnow() - timedelta(hours=2)
            expired_sessions = session.query(AppSession).filter(AppSession.last_activity < expiration_time).all()

            for expired_session in expired_sessions:
                session.delete(expired_session)

            session.commit()
            logger.info(f"Wyczyszczono {len(expired_sessions)} wygasłych sesji")
    except Exception as e:
        logger.error(f"Błąd w cleanup_expired_sessions: {str(e)}")


def generate_reflink_code():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))

@app.route('/generate_reflink', methods=['POST'])
def generate_reflink():
    data = request.get_json()
    user_id = data['user_id']

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User does not exist"}), 400

    reflink_code = generate_reflink_code()
    referral_link = ReferralLink(user_id=user.id, reflink_code=reflink_code)
    db.session.add(referral_link)
    db.session.commit()

    return jsonify({
        "user_id": user.id,
        "reflink_code": reflink_code
    }), 201

@app.route('/handle_referral', methods=['POST'])
def handle_referral():
    data = request.get_json()
    telegram_id = data.get('telegram_id')
    referrer_code = data.get('referrer_code')

    # Sprawdź, czy użytkownik już istnieje
    user = User.query.filter_by(telegram_id=telegram_id).first()
    if user:
        return jsonify({"message": "User already exists"}), 400

    # Dodaj nowego użytkownika
    new_user = User(
        telegram_id=telegram_id,
        username=data.get('username'),
        email=data.get('email')
    )
    db.session.add(new_user)
    db.session.commit()

    # Jeśli jest podany referrer_code, przypisz zaproszenie
    if referrer_code:
        referrer_link = ReferralLink.query.filter_by(reflink_code=referrer_code).first()
        if referrer_link:
            referrer_link.user_id = new_user.id
            db.session.commit()

    return jsonify({
        "user_id": new_user.id,
        "telegram_id": new_user.telegram_id
    }), 201

@app.route('/accept_invitation', methods=['GET'])
def accept_invitation():
    referrer_code = request.args.get('referrer_code')
    telegram_id = request.args.get('telegram_id')

    if not referrer_code or not telegram_id:
        return jsonify({"error": "Brak referrer_code lub telegram_id"}), 400

    user = User.query.filter_by(telegram_id=telegram_id).first()

    if not user:
        return jsonify({"message": "Użytkownik nie znaleziony, proszę zarejestruj się najpierw"}), 400

    referrer_link = ReferralLink.query.filter_by(reflink_code=referrer_code).first()

    if not referrer_link:
        return jsonify({"error": "Nieprawidłowy referrer_code"}), 400

    referrer_user = User.query.get(referrer_link.user_id)

    if not referrer_user:
        return jsonify({"error": "Referrer user not found"}), 400

    # Możesz dodać kod do nagradzania referrera tutaj, jeśli to potrzebne

    return jsonify({
        "message": "Zaproszenie zostało zaakceptowane!",
        "referrer_user": {
            "id": referrer_user.id,
            "telegram_id": referrer_user.telegram_id
        }
    }), 200

#Init data telegram i start aplikacji
def prepare_init_data(message):
    # Parse parameters from the message
    hash_params = urlparse(message.text).fragment
    params = parse_qs(hash_params)

    tg_web_app_data = json.loads(params.get('#tgWebAppData', [None])[0]) if params.get('#tgWebAppData') else {}
    token = params.get('token', [None])[0]
    session_id = params.get('session_id', [None])[0]
    query_id = params.get('#tgWebAppData', [None])[0]

    init_data = {
        'user_id': str(message.chat.id),
        'username': message.from_user.username,
        'first_name': message.from_user.first_name,
        'last_name': message.from_user.last_name,
        'language_code': message.from_user.language_code,
        'tgWebAppVersion': params.get('tgWebAppVersion', [None])[0],
        'tgWebAppData': tg_web_app_data,
        'tgWebAppPlatform': params.get('tgWebAppPlatform', [None])[0],
        'tgWebAppThemeParams': json.loads(params.get('tgWebAppThemeParams', [None])[0]) if params.get('tgWebAppThemeParams') else None,
        'tgWebAppStartParam': params.get('tgWebAppStartParam', [None])[0],
        'telegram_id': str(message.chat.id),
        'session_id': session_id,
        'token': token,
    }
    return init_data

def prepare_markup(session_id):
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("Play!", callback_data=f"play_{session_id}"),
        types.InlineKeyboardButton("Statystyki", callback_data="statystyki"),
        types.InlineKeyboardButton("Pomoc", callback_data="help"),
        types.InlineKeyboardButton("Facebook", url="https://www.facebook.com/yeticoiin"),
        types.InlineKeyboardButton("YetiCoin", url="https://www.yeticoin.pl"),
        types.InlineKeyboardButton("YetiAI", url="https://yetiai.pl")
    )
    return markup

@bot.callback_query_handler(func=lambda call: call is not None and call.data is not None)
def callback_query(call):
    logger.info(f"Otrzymano zapytanie zwrotne: {call.data} od użytkownika {call.from_user.id}")

    try:
        if call.data.startswith('play_'):
            with app.app_context():
                logger.debug("Wejście do kontekstu aplikacji")

                user_id = call.from_user.id
                logger.debug(f"ID użytkownika do pobrania sesji: {user_id}")

                existing_session = AppSession.query.filter_by(user_id=str(user_id)).order_by(AppSession.start_time.desc()).first()

                if existing_session:
                    session_id = existing_session.session_id
                    logger.debug(f"Pobrano istniejące session_id: {session_id}")

                    existing_session.start_time = datetime.utcnow()
                    db.session.commit()

                    logger.info(f"Pomyślnie zaktualizowano sesję dla użytkownika {user_id}")
                    handle_play_button(call, session_id)
                else:
                    logger.error(f"Nie znaleziono istniejącej sesji dla użytkownika {user_id}")
                    bot.answer_callback_query(call.id, "Nie znaleziono aktywnej sesji. Proszę rozpocząć grę ponownie używając /start.")

        elif call.data == "help":
            try:
                bot.answer_callback_query(call.id, "Wysyłanie pomocy...")
                help_message = ("🆘 Pomoc YetiCoinTap 🆘\n\n"
                                "Dostępne komendy:\n"
                                "/start - Rozpocznij grę\n"
                                "Aby rozpocząć grę, naciśnij przycisk 'Graj!' po wpisaniu /start.")
                sent_message = bot.send_message(call.message.chat.id, help_message)
                logger.info(f"Wysłano wiadomość pomocy: ID wiadomości {sent_message.message_id}")
            except Exception as e:
                logger.error(f"Error sending help message: {str(e)}", exc_info=True)

        elif call.data == "statystyki":
            try:
                bot.answer_callback_query(call.id, "Sprawdzam statystyki...")
                with app.app_context():
                    user = User.query.filter_by(telegram_id=str(call.from_user.id)).first()
                    if user:
                        progress = UserProgress.query.filter_by(user_id=user.telegram_id).first()
                        if progress:
                            stats_message = (f"Twoje statystyki:\n\n"
                                             f"Poziom: {progress.level}\n"
                                             f"YetiCoiny: {progress.coins}\n"
                                             f"Doświadczenie: {progress.experience}/100")
                            sent_message = bot.send_message(call.message.chat.id, stats_message)
                            logger.info(f"Wysłano wiadomość ze statystykami: ID wiadomości {sent_message.message_id}")
                        else:
                            bot.send_message(call.message.chat.id, "Nie znaleziono statystyk dla tego użytkownika.")
                            logger.warning(f"Nie znaleziono postępu dla użytkownika o ID: {user.telegram_id}")
                    else:
                        bot.send_message(call.message.chat.id, "Nie znaleziono użytkownika. Rozpocznij grę, aby utworzyć profil.")
                        logger.warning(f"Nie znaleziono użytkownika o Telegram ID: {call.from_user.id}")
            except Exception as e:
                logger.error(f"Error fetching statistics: {str(e)}", exc_info=True)

        else:
            logger.warning(f"Nieznane zapytanie zwrotne: {call.data}")
            bot.answer_callback_query(call.id, "Nieznane polecenie")

    except Exception as e:
        logger.error(f"Błąd w callback_query: {str(e)}", exc_info=True)
        bot.answer_callback_query(call.id, "Wystąpił błąd podczas przetwarzania żądania")


# Uruchomienie aplikacji Flask w głównym wątku
if __name__ == '__main__':
    passive_income_thread = Thread(target=calculate_passive_income)
    passive_income_thread.daemon = True
    passive_income_thread.start()
    WEBHOOK_URL = f"{GAME_URL}/webhook"

    set_webhook(WEBHOOK_URL)
    app.run(debug=True, use_reloader=True, port=int(os.getenv('PORT', 5000)))