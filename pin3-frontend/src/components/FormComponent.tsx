import React, { useState, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';
import { SelectAiModel } from './SelectAiModel';
import { SelectImage } from './SelectImage';

interface FormData {
    aiModel: string;
    image: string | null;
}

const FormComponent: React.FC = () => {
    const [formData, setFormData] = useState<FormData>({
        aiModel: 'pytorch',
        image: null,
    });

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
            const response = await axios.post('http://localhost:5000/api/data', formData);
            console.log(response.data);
            // Handle success
        } catch (error) {
            console.error('Error:', error);
            // Handle error
        }
    };

    return (
        <form onSubmit={handleSubmit} className="inner-container">
            <SelectAiModel value={formData.aiModel} onChange={handleModelChange} />
            <SelectImage onImageChange={handleImageChange} />
            <button type="submit">Submit</button>
        </form>
    );
};

export default FormComponent;
