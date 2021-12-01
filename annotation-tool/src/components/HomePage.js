
import datasetPic from '../assets/datasetPic.png'
import './HomePage.css'

import React from 'react'

const HomePage = () => {
    return (
        
        <>
            <section className="welcomeSec ">
                
                <div >
                    <h1 >The tool to train your datasets</h1>
                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. </p>
                    <div className="btnContainer">
                        <button>Get started</button>
                    </div>
                </div>
                <img src={datasetPic} alt="dataset" />
                
                
            </section>
            <section className="featuresSec">
                <h1>sec 222</h1>
            </section>
        </>
        
    )
}

export default HomePage
