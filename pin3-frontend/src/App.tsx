import {SelectAiModel} from "./components/SelectAiModel.tsx";
import {ConfirmationButton} from "./components/ConfirmationButton.tsx";
import './style.css';
import {SelectImage} from "./components/SelectImage.tsx";

function App() {
  return <main className={'outer-container'}>
    <form className={'inner-container'}>
      <SelectAiModel/>
      <SelectImage/>
      <ConfirmationButton/>
    </form>
  </main>
}

export default App
