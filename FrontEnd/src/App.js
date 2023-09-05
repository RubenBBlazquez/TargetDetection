import React from "react";
import {BrowserRouter, Routes, Route} from 'react-router-dom';
import Predictions from "./Predictions/Predictions";
import SideBar from "./CommonComponents/SideBar";
import {Training} from "./Training/Training";

function App() {
    return (
        <div>
            <BrowserRouter>
                <div className={'d-flex'}>
                    <SideBar/>

                    <div className={'w-100'}>
                        <Routes>
                            <Route path="/" element={<Predictions/>}/>
                            <Route path="/predictions" element={<Predictions/>}/>
                            <Route path="/training" element={<Training/>}/>
                        </Routes>
                    </div>

                </div>
            </BrowserRouter>
        </div>

    );
}

export default App;