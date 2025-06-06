from app import create_app, bcrypt
from app.db import init_db, create_admin_user

app = create_app()

if __name__ == '__main__':
    create_admin_user(bcrypt)
    init_db()
    app.run(debug=True)
