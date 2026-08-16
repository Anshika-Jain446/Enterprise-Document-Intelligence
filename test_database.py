from database import Database

db = Database()

connection = db.get_connection()

print("PostgreSQL connection successful!")

connection.close()