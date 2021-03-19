from app import app, measure
from core.CosinMeasure import CosineMeasure

if __name__ == "__main__":
    measure = CosineMeasure()
    measure.prepare_dataset()
    app.run()