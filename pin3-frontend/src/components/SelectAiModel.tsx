const PY_TORCH = 'pytorch';
const FAST_AI = 'fastai';

export function SelectAiModel() {
    return (
        <select id={'aiSelect'} name={'aiSelect'}>
            <option value={PY_TORCH}>PyTorch</option>
            <option value={FAST_AI}>FastAI</option>
        </select>
    );
}