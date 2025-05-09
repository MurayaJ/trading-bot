import sqlite3
from datetime import datetime

DB_PATH = "trading_bot_data/db/trading_users.db"

def manage_user(action, email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if action == "add":
        name = input("Enter user name: ")
        c.execute("INSERT INTO users (name, email, subscription_status, trading_status) VALUES (?, ?, 'active', 'inactive')", (name, email))
    elif action == "activate":
        c.execute("UPDATE users SET subscription_status = 'active', last_payment_date = ? WHERE email = ?", (datetime.now().isoformat(), email))
    elif action == "deactivate":
        c.execute("UPDATE users SET subscription_status = 'expired' WHERE email = ?", (email,))
    elif action == "block":
        c.execute("UPDATE users SET subscription_status = 'blocked' WHERE email = ?", (email,))
    conn.commit()
    conn.close()
    print(f"User {email} {action}ed successfully!")

if __name__ == "__main__":
    action = input("Enter action (add/activate/deactivate/block): ")
    email = input("Enter user email: ")
    manage_user(action, email)