import {BrowserRouter, Routes, Route} from 'react-router-dom';
import Predictions from "./Predictions/Predictions";
import SideBar from "./Components/SideBar";

function App() {
    return (
        <div>
            <BrowserRouter>
                <div className={'d-flex'}>
                    <SideBar/>

                    <div className={'w-100'}>
                        <Routes>
                            <Route path="/predictions" element={<Predictions/>}/>
                            <Route path="/training" element={<Predictions/>}/>
                        </Routes>
                    </div>

                </div>
            </BrowserRouter>
        </div>

    );
}

export default App;