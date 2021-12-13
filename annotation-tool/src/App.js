
import './App.css';
import {BrowserRouter as Router, Routes, Route} from 'react-router-dom';
import { useState } from 'react';
//import { useStateWithCallbackLazy } from 'use-state-with-callback';

import Navbar from './components/Navbar';
import HomePage from './components/HomePage';
import Footer from './components/Footer';
import AppBody from './components/AppBody';


function App() {
  const [datasets, setDatasets] = useState([]); 
  //const [test, setTest] = useStateWithCallbackLazy(0);

  
  return (
    <div className="App">
      <Navbar/>

      <Router>
        <Routes>
            <Route path="/" element= {<HomePage/>} />
            <Route path="/app" element= {<AppBody datasets={datasets} setData={setDatasets} />} />
            <Route path="*" element={<Nomatch/>}/>
        </Routes>

      </Router>
      <Footer/>

    </div>
  );
}   

export default App;




// default page when a false url is entred

function Nomatch() {
  return (
    <div>
      <section style={ NomatchStyle }>Ops!!! rong URL :( </section>
    </div>
  )
}

const NomatchStyle = {
  "font-size": "50px",
  "font-weight" : "bold",
  "height": "100vh",
  "display": "flex",
  "justify-content" : "center",
  "align-items" : "center"
}