from app import app, measure
from core.SemanticMeasure import SemanticMeasure

if __name__ == "__main__":
    measure = SemanticMeasure(verbose=True)
    app.run()