
import datasetPic from '../assets/datasetPic.png'
import dataEdit from '../assets/dataEdit.png'
import docView from '../assets/docView.png'
import docAnnot from '../assets/docAnnot.png'

import './HomePage.css'

import React from 'react'
import { Link } from 'react-router-dom'

const HomePage = () => {
    return (
        
        <>
            <section className="welcomeSec ">
                
                <div className="titleHolder">
                    <h1 >The tool to train your datasets</h1>
                    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam. </p>
                    <button>Get started</button>
                    
                </div>
                <img src={datasetPic} alt="dataset"/>    
            </section>
            <section className="featuresSec">
                <h1>Our features</h1>
                <div className="featuresCont">
                    <div className="featureCard">
                        <img src={dataEdit} alt="data Edit feature" />
                        <p>
                            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
                        </p>
                    </div>
                    <div className="featureCard">
                        <img src={docView} alt="data Edit feature" />
                        <p>
                            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                        </p>
                    </div>
                    <div className="featureCard">
                        <img src={docAnnot} alt="data Edit feature" />
                        <p>
                            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                        </p>
                    </div>
                </div>

            </section>
            
        </>
        
    )
}

export default HomePage
