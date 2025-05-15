from app import create_app, bcrypt
from app.db import init_db, create_admin_user

app = create_app()

if __name__ == '__main__':
    # Inicializiraj bazo in ustvari admin uporabnika, če še ne obstaja
    init_db()
    create_admin_user(bcrypt)

    app.run(debug=True)
