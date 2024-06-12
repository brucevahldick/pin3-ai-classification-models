import React, {useState, ChangeEvent, FormEvent} from 'react';
import axios from 'axios';
import {SelectAiModel} from './SelectAiModel';
import {SelectImage} from './SelectImage';
import PredictionDisplay from './PredictionDisplay';

interface FormData {
    aiModel: string;
    image: string | null;
}

const FormComponent: React.FC = () => {
    const [formData, setFormData] = useState<FormData>({
        aiModel: 'pytorch',
        image: null,
    });
    const [prediction, setPrediction] = useState<string | null>(null);

    const handleModelChange = (e: ChangeEvent<HTMLSelectElement>) => {
        setFormData((prevData) => ({
            ...prevData,
            aiModel: e.target.value,
        }));
    };

    const handleImageChange = (image: string | null) => {
        setFormData((prevData) => ({
            ...prevData,
            image,
        }));
    };

    const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        try {
            const response = await axios.post<{ prediction: string }>('http://localhost:5000/api/data', formData);
            console.log(response.data);
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error('Error:', error);
            // Handle error
        }
    };

    return (
        <div className="form-container">
            <form onSubmit={handleSubmit} className="inner-container">
                <SelectAiModel value={formData.aiModel} onChange={handleModelChange}/>
                <SelectImage onImageChange={handleImageChange}/>
                {prediction && <PredictionDisplay prediction={prediction}/>}
                <button type="submit">Classificar</button>
            </form>
        </div>
    );
};

export default FormComponent;
