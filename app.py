
import streamlit as st
import threading
import time
import websocket
import json
import numpy as np
import pandas as pd
import talib
from datetime import datetime
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sqlite3
import bcrypt

# Define model directory and database path
MODEL_DIR = "/content/drive/MyDrive/trading_bot_models"
DB_PATH = "/content/drive/MyDrive/trading_bot_users.db"
os.makedirs(MODEL_DIR, exist_ok=True)

# Database functions
def init_db():
    """Initialize the SQLite database with all required columns."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            trial_start_date TEXT,
            subscription_status TEXT,
            last_payment_date TEXT
        )
    """)
    c.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in c.fetchall()]
    if "password_hash" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    if "trial_start_date" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN trial_start_date TEXT")
    if "subscription_status" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN subscription_status TEXT")
    if "last_payment_date" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN last_payment_date TEXT")
    conn.commit()
    conn.close()

def register_user(name, email, password):
    """Register a new user with trial period."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        trial_start_date = datetime.now().isoformat()
        subscription_status = 'trial'
        c.execute("""
            INSERT INTO users (name, email, password_hash, trial_start_date, subscription_status)
            VALUES (?, ?, ?, ?, ?)
        """, (name, email, password_hash, trial_start_date, subscription_status))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Email already exists
    except Exception as e:
        st.error(f"Registration error: {e}")
        return False
    finally:
        conn.close()

def login_user(email, password):
    """Authenticate a user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("SELECT password_hash FROM users WHERE email = ?", (email,))
        result = c.fetchone()
        if result and bcrypt.checkpw(password.encode("utf-8"), result[0]):
            return True
        return False
    except Exception as e:
        st.error(f"Login error: {e}")
        return False
    finally:
        conn.close()

def get_user_data(email):
    """Fetch user data."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""
            SELECT id, name, email, password_hash, trial_start_date, subscription_status, last_payment_date
            FROM users WHERE email = ?
        """, (email,))
        row = c.fetchone()
        if row:
            return {
                'id': row[0], 'name': row[1], 'email': row[2], 'password_hash': row[3],
                'trial_start_date': row[4], 'subscription_status': row[5], 'last_payment_date': row[6]
            }
        return None
    except Exception as e:
        st.error(f"Database error: {e}")
        return None
    finally:
        conn.close()

def check_subscription_status(email):
    """Check and update subscription status."""
    user_data = get_user_data(email)
    if not user_data:
        return None
    try:
        if user_data['subscription_status'] == 'trial':
            if user_data['trial_start_date']:
                trial_start = datetime.fromisoformat(user_data['trial_start_date'])
                if (datetime.now() - trial_start).days > 7:
                    update_user_status(email, 'expired')
                    user_data['subscription_status'] = 'expired'
            else:
                trial_start_date = datetime.now().isoformat()
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute("UPDATE users SET trial_start_date = ? WHERE email = ?", (trial_start_date, email))
                conn.commit()
                conn.close()
                user_data['trial_start_date'] = trial_start_date
        elif user_data['subscription_status'] == 'active':
            if user_data['last_payment_date']:
                last_payment = datetime.fromisoformat(user_data['last_payment_date'])
                if (datetime.now() - last_payment).days > 30:
                    update_user_status(email, 'expired')
                    user_data['subscription_status'] = 'expired'
            else:
                update_user_status(email, 'expired')
                user_data['subscription_status'] = 'expired'
        return user_data
    except Exception as e:
        st.error(f"Subscription check error: {e}")
        return user_data

def update_user_status(email, status):
    """Update user's subscription status."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("UPDATE users SET subscription_status = ? WHERE email = ?", (status, email))
        conn.commit()
    except Exception as e:
        st.error(f"Status update error: {e}")
    finally:
        conn.close()

# Trading bot class
class TradingBot:
    def __init__(self, app_id, token, target_profit, session_id):
        """Initialize the trading bot."""
        self.app_id = app_id
        self.token = token
        self.target_profit = float(target_profit)
        self.session_id = session_id
        self.account_balance = 1000.00
        self.initial_amount = 0.35
        self.amount = self.initial_amount
        self.price = self.initial_amount
        self.cumulative_profit = 0.00
        self.is_trading = False
        self.last_trade_time = 0
        self.trade_cooldown = 5
        self.entry_tick = None
        self.last_features = None
        self.prediction_tick = None
        self.consecutive_losses = 0
        self.recent_trades = []
        self.digit_history = []
        self.training_samples = 0
        self.output = []
        self.stop_trading = False

        # Machine learning parameters
        self.alpha_win = 0.15
        self.alpha_loss = 0.05
        self.markov_p1 = np.full((10, 10), 0.1)
        self.markov_p2 = np.full((100, 10), 0.1)
        self.rf_digit_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_scaler = StandardScaler()
        self.expected_features = 10
        self.training_data = []
        self.training_targets = []
        self.training_weights = []

        # DataFrame for indicators
        self.df = pd.DataFrame(columns=["Time", "Tick", "Last_Digit", "MA_6", "RSI", "Volatility", "Hour_sin", "Hour_cos"]).astype({
            "Time": "datetime64[ns]", "Tick": "float64", "Last_Digit": "int64", "MA_6": "float64",
            "RSI": "float64", "Volatility": "float64", "Hour_sin": "float64", "Hour_cos": "float64"
        })

        self.load_models()

    def get_last_digit(self, tick):
        """Extract the last digit from a tick."""
        try:
            tick_rounded = round(float(tick), 2)
            return int(f"{tick_rounded:.2f}"[-1])
        except Exception:
            return 0

    def load_models(self):
        """Load machine learning models."""
        try:
            self.markov_p1 = joblib.load(os.path.join(MODEL_DIR, f"markov_p1_{self.session_id}.joblib"))
            self.markov_p2 = joblib.load(os.path.join(MODEL_DIR, f"markov_p2_{self.session_id}.joblib"))
            self.rf_digit_predictor = joblib.load(os.path.join(MODEL_DIR, f"rf_digit_predictor_{self.session_id}.joblib"))
            self.feature_scaler = joblib.load(os.path.join(MODEL_DIR, f"feature_scaler_{self.session_id}.joblib"))
            if self.markov_p1.shape != (10, 10) or self.markov_p2.shape != (100, 10):
                self.markov_p1 = np.full((10, 10), 0.1)
                self.markov_p2 = np.full((100, 10), 0.1)
        except Exception:
            self.markov_p1 = np.full((10, 10), 0.1)
            self.markov_p2 = np.full((100, 10), 0.1)
            self.rf_digit_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
            self.feature_scaler = StandardScaler()

    def save_models(self):
        """Save machine learning models."""
        try:
            joblib.dump(self.markov_p1, os.path.join(MODEL_DIR, f"markov_p1_{self.session_id}.joblib"))
            joblib.dump(self.markov_p2, os.path.join(MODEL_DIR, f"markov_p2_{self.session_id}.joblib"))
            joblib.dump(self.rf_digit_predictor, os.path.join(MODEL_DIR, f"rf_digit_predictor_{self.session_id}.joblib"))
            joblib.dump(self.feature_scaler, os.path.join(MODEL_DIR, f"feature_scaler_{self.session_id}.joblib"))
        except Exception as e:
            self.output.append(f"Error saving models: {e}")
            st.session_state[f"output_{self.session_id}"] = self.output

    def calculate_profit(self, is_win):
        """Calculate profit or loss."""
        profit = round(self.amount * 0.8857, 2) if is_win else -round(self.amount, 2)
        return profit

    def adjust_amount(self, is_win):
        """Adjust trading amount."""
        if is_win:
            self.amount = self.initial_amount
            self.consecutive_losses = 0
        else:
            self.amount = min(round(self.amount * 2.2, 2), max(self.account_balance * 0.9, 0))
            self.consecutive_losses += 1
        self.price = self.amount

    def update_dataframe(self, tick, timestamp):
        """Update DataFrame with indicators."""
        try:
            tick = float(tick)
            last_digit = self.get_last_digit(tick)
            new_row = {
                "Time": timestamp, "Tick": tick, "Last_Digit": last_digit,
                "Hour_sin": np.sin(2 * np.pi * timestamp.hour / 24),
                "Hour_cos": np.cos(2 * np.pi * timestamp.hour / 24)
            }
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            if len(self.df) >= 6:
                self.df["MA_6"] = talib.SMA(self.df["Tick"].values, timeperiod=6)
            if len(self.df) >= 14:
                self.df["RSI"] = talib.RSI(self.df["Tick"].values, timeperiod=14)
            if len(self.df) >= 20:
                self.df["Volatility"] = self.df["Tick"].rolling(window=20).std()
            self.df = self.df.tail(100)
        except Exception as e:
            self.output.append(f"DataFrame update error: {e}")

    def update_markov_models(self, d_t, d_t_plus_1, d_t_minus_1=None, is_win=None):
        """Update Markov models."""
        try:
            alpha = self.alpha_win if is_win else self.alpha_loss if is_win is not None else 0.1
            adjustment = 1.5 if is_win else 0.5 if is_win is not None else 1.0
            self.markov_p1[d_t, d_t_plus_1] = (1 - alpha) * self.markov_p1[d_t, d_t_plus_1] + alpha * adjustment
            for k in range(10):
                if k != d_t_plus_1:
                    self.markov_p1[d_t, k] = (1 - alpha) * self.markov_p1[d_t, k]
            self.markov_p1[d_t, :] /= np.sum(self.markov_p1[d_t, :])
            if d_t_minus_1 is not None:
                state = 10 * d_t_minus_1 + d_t
                self.markov_p2[state, d_t_plus_1] = (1 - alpha) * self.markov_p2[state, d_t_plus_1] + alpha * adjustment
                for k in range(10):
                    if k != d_t_plus_1:
                        self.markov_p2[state, k] = (1 - alpha) * self.markov_p2[state, k]
                self.markov_p2[state, :] /= np.sum(self.markov_p2[state, :])
        except Exception as e:
            self.output.append(f"Markov update error: {e}")

    def get_features(self):
        """Extract features."""
        if len(self.digit_history) < 3 or len(self.df) < 20:
            return None
        recent_digits = self.digit_history[-3:]
        d_t = self.digit_history[-1]
        d_t_minus_1 = self.digit_history[-2] if len(self.digit_history) >= 2 else 0
        P_even, P_odd = self.predict_one_step(d_t, d_t_minus_1)
        indicators = self.df.iloc[-1][["MA_6", "RSI", "Volatility", "Hour_sin", "Hour_cos"]].values
        features = list(recent_digits) + [P_even, P_odd] + list(indicators)
        return features if len(features) == self.expected_features else None

    def train_rf_predictor(self):
        """Train Random Forest predictor."""
        if len(self.training_data) < 100:
            return
        try:
            X = np.array(self.training_data)
            y = np.array(self.training_targets)
            weights = np.array(self.training_weights)
            if X.shape[1] != self.expected_features:
                self.rf_digit_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
                self.feature_scaler = StandardScaler()
                return
            self.feature_scaler.fit(X)
            X_scaled = self.feature_scaler.transform(X)
            self.rf_digit_predictor.fit(X_scaled, y, sample_weight=weights)
        except Exception as e:
            self.output.append(f"RF training error: {e}")

    def predict_one_step(self, d_t, d_t_minus_1=None):
        """Predict next digit probabilities."""
        try:
            if d_t_minus_1 is not None:
                p_next = 0.5 * self.markov_p1[d_t, :] + 0.5 * self.markov_p2[10 * d_t_minus_1 + d_t, :]
            else:
                p_next = self.markov_p1[d_t, :]
            P_even = np.sum(p_next[[0, 2, 4, 6, 8]])
            P_odd = np.sum(p_next[[1, 3, 5, 7, 9]])
            return P_even, P_odd
        except Exception:
            return 0.5, 0.5

    def get_rf_prediction(self, features):
        """Get RF prediction."""
        if features is None or not hasattr(self.feature_scaler, "n_features_in_"):
            return [0.5, 0.5]
        try:
            features_scaled = self.feature_scaler.transform([features])
            pred_proba = self.rf_digit_predictor.predict_proba(features_scaled)[0]
            return pred_proba if len(pred_proba) == 2 else [0.5, 0.5]
        except Exception:
            return [0.5, 0.5]

    def get_rf_threshold(self):
        """Calculate trading threshold."""
        base_threshold = 0.55
        loss_increment = 0.02
        max_threshold = 0.75
        threshold = base_threshold + loss_increment * min(self.consecutive_losses, 3) + (0.05 * max(self.consecutive_losses - 3, 0))
        return min(threshold, max_threshold)

    def buy_contract(self, ws, contract_type):
        """Place a trading contract."""
        try:
            self.last_features = self.get_features()
            self.prediction_tick = self.digit_history[-2] if len(self.digit_history) >= 2 else None
            self.entry_tick = self.digit_history[-1]
            self.is_trading = True
            self.last_trade_time = time.time()
            json_data = json.dumps({
                "buy": 1, "subscribe": 1, "price": round(self.price, 2),
                "parameters": {
                    "amount": round(self.amount, 2), "basis": "stake", "contract_type": contract_type,
                    "currency": "USD", "duration": 1, "duration_unit": "t", "symbol": "R_100"
                }
            })
            self.output.append(f"Trade placed: {contract_type}, Entry: {self.entry_tick}, Amount: {self.amount:.2f}")
            st.session_state[f"output_{self.session_id}"] = self.output
            ws.send(json_data)
        except Exception as e:
            self.output.append(f"Buy contract error: {e}")

    def on_open(self, ws):
        """Handle WebSocket open."""
        try:
            ws.send(json.dumps({"authorize": self.token}))
        except Exception as e:
            self.output.append(f"WebSocket open error: {e}")

    def on_message(self, ws, message):
        """Handle WebSocket messages."""
        try:
            data = json.loads(message)
            if "error" in data:
                self.output.append(f"API Error: {data['error']['message']}")
                st.session_state[f"output_{self.session_id}"] = self.output
                return
            if data["msg_type"] == "authorize":
                ws.send(json.dumps({"ticks": "R_100", "subscribe": 1}))
            elif data["msg_type"] == "tick":
                tick = float(data["tick"]["quote"])
                timestamp = datetime.fromtimestamp(data["tick"]["epoch"])
                last_digit = self.get_last_digit(tick)
                if len(self.digit_history) >= 3:
                    features = self.get_features()
                    if features:
                        target = 0 if last_digit % 2 == 0 else 1
                        self.training_data.append(features)
                        self.training_targets.append(target)
                        self.training_weights.append(1.0)
                        if len(self.training_data) > 1000:
                            self.training_data.pop(0)
                            self.training_targets.pop(0)
                            self.training_weights.pop(0)
                self.digit_history.append(last_digit)
                self.update_dataframe(tick, timestamp)
                if len(self.digit_history) >= 2:
                    d_t = self.digit_history[-2]
                    d_t_plus_1 = self.digit_history[-1]
                    d_t_minus_1 = self.digit_history[-3] if len(self.digit_history) >= 3 else None
                    self.update_markov_models(d_t, d_t_plus_1, d_t_minus_1)
                    self.training_samples += 1
                    if self.training_samples % 100 == 0:
                        self.save_models()
                        self.train_rf_predictor()
                if (self.training_samples >= 100 and not self.is_trading and
                    (time.time() - self.last_trade_time) >= self.trade_cooldown and
                    self.cumulative_profit < self.target_profit and not self.stop_trading and
                    self.account_balance > 0):
                    features = self.get_features()
                    if features:
                        rf_pred = self.get_rf_prediction(features)
                        d_t = self.digit_history[-1]
                        d_t_minus_1 = self.digit_history[-2] if len(self.digit_history) >= 2 else None
                        P_even, P_odd = self.predict_one_step(d_t, d_t_minus_1)
                        rf_threshold = self.get_rf_threshold()
                        if rf_pred[0] > rf_threshold and P_even > 0.60:
                            self.buy_contract(ws, "DIGITEVEN")
                        elif rf_pred[1] > rf_threshold and P_odd > 0.60:
                            self.buy_contract(ws, "DIGITODD")
            elif "proposal_open_contract" in data:
                contract = data["proposal_open_contract"]
                if contract.get("is_sold", False):
                    exit_tick = float(contract["exit_tick"])
                    last_digit = self.get_last_digit(exit_tick)
                    contract_type = contract["contract_type"]
                    is_win = (contract_type == "DIGITEVEN" and last_digit % 2 == 0) or                              (contract_type == "DIGITODD" and last_digit % 2 != 0)
                    profit = self.calculate_profit(is_win)
                    self.adjust_amount(is_win)
                    self.account_balance = max(self.account_balance + profit, 0)
                    self.cumulative_profit += profit
                    self.recent_trades.append((contract_type, is_win))
                    if len(self.recent_trades) > 50:
                        self.recent_trades.pop(0)
                    if self.prediction_tick is not None and self.entry_tick is not None and self.last_features:
                        self.update_markov_models(self.prediction_tick, self.entry_tick, self.digit_history[-3] if len(self.digit_history) >= 3 else None)
                        self.update_markov_models(self.entry_tick, last_digit, self.prediction_tick, is_win=is_win)
                        target = 0 if last_digit % 2 == 0 else 1
                        weight = 2.0 if is_win else 0.5
                        self.training_data.append(self.last_features)
                        self.training_targets.append(target)
                        self.training_weights.append(weight)
                        self.train_rf_predictor()
                    self.output.append(f"Result: {contract_type}, Entry: {self.entry_tick}, Exit: {exit_tick:.2f}, {'Win' if is_win else 'Loss'}, Profit: {profit:.2f}, Balance: {self.account_balance:.2f}")
                    st.session_state[f"output_{self.session_id}"] = self.output
                    self.is_trading = False
                    self.entry_tick = None
                    self.last_features = None
                    self.prediction_tick = None
                    if self.cumulative_profit >= self.target_profit or self.account_balance <= 0:
                        self.save_models()
                        self.stop_trading = True
                        ws.close()
        except Exception as e:
            self.output.append(f"Message handling error: {e}")
            st.session_state[f"output_{self.session_id}"] = self.output

    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        self.output.append(f"WebSocket Error: {error}")
        st.session_state[f"output_{self.session_id}"] = self.output

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        self.output.append(f"WebSocket closed: {close_msg}")
        st.session_state[f"output_{self.session_id}"] = self.output

    def run(self):
        """Run the trading bot with reconnection."""
        while not self.stop_trading and self.cumulative_profit < self.target_profit and self.account_balance > 0:
            try:
                api_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
                ws = websocket.WebSocketApp(
                    api_url, on_message=self.on_message, on_open=self.on_open,
                    on_error=self.on_error, on_close=self.on_close
                )
                ws.run_forever(ping_interval=20, ping_timeout=10)
                if self.stop_trading or self.cumulative_profit >= self.target_profit:
                    self.save_models()
                    break
                self.output.append("Reconnecting in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                self.output.append(f"Run error: {e}")
                st.session_state[f"output_{self.session_id}"] = self.output
                time.sleep(5)

# Streamlit Interface
def main():
    """Main Streamlit interface."""
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .stTextInput>div>input, .stNumberInput>div>input {
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        .stMetric {
            font-size: 18px;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    init_db()
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "email" not in st.session_state:
        st.session_state["email"] = None
    if "last_activity" not in st.session_state:
        st.session_state["last_activity"] = time.time()
    if "trading_active" not in st.session_state:
        st.session_state["trading_active"] = False

    if not st.session_state["logged_in"]:
        st.title("Trading Bot")
        option = st.selectbox("Choose an option", ["Login", "Register"])
        if option == "Register":
            with st.form("register_form"):
                name = st.text_input("Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Register")
                if submit:
                    if name and email and password:
                        if register_user(name, email, password):
                            st.success("Registered successfully! Please log in.")
                        else:
                            st.error("Email already exists.")
                    else:
                        st.error("Please fill in all fields.")
        else:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                if submit:
                    if email and password:
                        if login_user(email, password):
                            st.session_state["logged_in"] = True
                            st.session_state["email"] = email
                            st.session_state["last_activity"] = time.time()
                            st.success("Logged in successfully!")
                            # Removed st.experimental_rerun()
                        else:
                            st.error("Invalid email or password.")
                    else:
                        st.error("Please enter email and password.")
    else:
        # Inactivity check (20 minutes)
        if time.time() - st.session_state["last_activity"] > 1200:
            if "bot" in st.session_state:
                st.session_state["bot"].stop_trading = True
            st.session_state["logged_in"] = False
            st.session_state["email"] = None
            st.session_state.pop("bot", None)
            st.session_state.pop("thread", None)
            st.session_state["trading_active"] = False
            st.warning("Logged out due to inactivity.")
            # Removed st.experimental_rerun()
            return

        # Subscription check
        user_data = check_subscription_status(st.session_state["email"])
        if not user_data:
            st.error("User not found.")
            st.session_state["logged_in"] = False
            return
        if user_data["subscription_status"] == "blocked":
            st.error("Your account is blocked.")
            st.session_state["logged_in"] = False
            return
        elif user_data["subscription_status"] == "expired":
            st.error("Your subscription has expired. Please renew to continue.")
            st.markdown("[Pay 70 GBP via Skrill](https://skrill.me/rq/John/70/GBP?key=p56pcU69FeFTB70NHi9Qh3Q2RQ8)")
            st.write("After payment, contact the administrator to activate your account.")
            return

        # Dashboard
        st.title("Trading Dashboard")
        st.write(f"Welcome, {user_data['name']} ({st.session_state['email']})")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Trading Controls")
            app_id = st.text_input("App ID", key=f"app_id_{st.session_state['email']}")
            token = st.text_input("Token", type="password", key=f"token_{st.session_state['email']}")
            target_profit = st.number_input("Target Profit ($)", min_value=0.01, value=10.0, step=0.1)
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if not st.session_state["trading_active"]:
                    if st.button("Start Trading"):
                        if app_id and token and target_profit > 0:
                            session_id = st.session_state["email"]
                            bot = TradingBot(app_id, token, target_profit, session_id)
                            st.session_state["bot"] = bot
                            st.session_state[f"output_{session_id}"] = []
                            st.session_state["trading_active"] = True
                            thread = threading.Thread(target=bot.run)
                            st.session_state["thread"] = thread
                            thread.start()
                            st.success("Trading started!")
                        else:
                            st.error("Please enter App ID, Token, and a valid Target Profit.")
            with col_btn2:
                if st.session_state["trading_active"]:
                    if st.button("Stop Trading"):
                        st.session_state["bot"].stop_trading = True
                        st.session_state["trading_active"] = False
                        st.success("Trading stopped.")
        with col2:
            st.subheader("Account Status")
            if "bot" in st.session_state:
                bot = st.session_state["bot"]
                st.metric("Balance", f"${bot.account_balance:.2f}")
                st.metric("Profit", f"${bot.cumulative_profit:.2f}")
            else:
                st.metric("Balance", "$1000.00")
                st.metric("Profit", "$0.00")
            if st.button("Logout"):
                if "bot" in st.session_state:
                    st.session_state["bot"].stop_trading = True
                st.session_state["logged_in"] = False
                st.session_state["email"] = None
                st.session_state.pop("bot", None)
                st.session_state.pop("thread", None)
                st.session_state["trading_active"] = False
                st.success("Logged out successfully!")
                # Removed st.experimental_rerun()

        st.subheader("Trade Output")
        session_id = st.session_state["email"]
        output_container = st.empty()
        if st.button("Refresh Output"):
            if f"output_{session_id}" in st.session_state:
                with output_container.container():
                    for line in st.session_state[f"output_{session_id}"]:
                        st.write(line)
        st.info("Click 'Refresh Output' to see the latest trading updates.")
        if "bot" in st.session_state and st.session_state["bot"].cumulative_profit >= st.session_state["bot"].target_profit:
            st.success(f"Target profit of ${st.session_state['bot'].target_profit:.2f} reached! Start a new session.")

        st.info("This app runs while the Colab notebook is active. For 24/7 use, deploy to a cloud service.")
        st.session_state["last_activity"] = time.time()

if __name__ == "__main__":
    main()
