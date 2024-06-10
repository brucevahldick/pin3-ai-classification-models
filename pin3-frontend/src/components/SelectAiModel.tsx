import React from "react";

const PY_TORCH = 'pytorch';
const FAST_AI = 'fastai';

interface SelectAiModelProps {
    value: string;
    onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
}

export const SelectAiModel: React.FC<SelectAiModelProps> = ({value, onChange}) => {
    return (
        <select id="aiSelect" name="aiSelect" value={value} onChange={onChange}>
            <option value={PY_TORCH}>PyTorch</option>
            <option value={FAST_AI}>FastAI</option>
        </select>
    );
}