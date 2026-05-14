from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class StudentPrediction(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    ca1 = db.Column(db.Float)

    ca2 = db.Column(db.Float)

    assignment = db.Column(db.Float)

    midterm = db.Column(db.Float)

    attendance = db.Column(db.Float)

    predicted_grade = db.Column(db.String(10))

    pass_fail = db.Column(db.String(10))

    risk_level = db.Column(db.String(20))

    confidence = db.Column(db.Float)

    def to_dict(self):

        return {

            "id": self.id,
            "ca1": self.ca1,
            "ca2": self.ca2,
            "assignment": self.assignment,
            "midterm": self.midterm,
            "attendance": self.attendance,
            "predicted_grade": self.predicted_grade,
            "pass_fail": self.pass_fail,
            "risk_level": self.risk_level,
            "confidence": self.confidence

        }