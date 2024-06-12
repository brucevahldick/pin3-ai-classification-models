import React from 'react';

interface PredictionProps {
    prediction: string;
}

const PredictionDisplay: React.FC<PredictionProps> = ({prediction}) => {
    return (
        <div className="prediction-container">
            <h2>Predição:</h2>
            <p>{prediction}</p>
        </div>
    );
};

export default PredictionDisplay;
