import React, {ChangeEvent, useState} from 'react';
import styled from 'styled-components';

const InputContainer = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
`;

const ImageInput = styled.input`
    display: none;
`;

const Label = styled.label`
    cursor: pointer;
    padding: 10px;
    background-color: #000;
    color: #fff;
    border-radius: 5px;
    border: 1px solid #fff;
    width: 100%;
    text-align: center;
`;

const ImagePreview = styled.img`
    margin-top: 20px;
    max-width: 100%;
`;

interface SelectImageProps {
    onImageChange: (image: string | null) => void;
}

export const SelectImage: React.FC<SelectImageProps> = ({onImageChange}) => {
    const [image, setImage] = useState<string | null>(null);

    const handleImageChange = (e: ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = () => {
                const result = reader.result as string;
                setImage(result);
                onImageChange(result);
            };
            reader.readAsDataURL(file);
        }
    };

    return (
        <InputContainer>
            <ImageInput
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                id="imageInput"
                name="imageInput"
            />
            <Label htmlFor="imageInput">Choose Image</Label>
            {image && <ImagePreview src={image} alt="Preview"/>}
        </InputContainer>
    );
};
