import React from 'react'
import { useState } from 'react/cjs/react.development';
import './DocView.css';
import arrowIcon from '../assets/arrow-white.svg';
import reactStringReplace from 'react-string-replace'
import Word from './Word';

import Menu from './Menu';





const DocView = ({text, id, texts, setTexts, datasets, colors}) => {
    const [collapsed, setCollapsed] = useState(true);
    //var parse = require('html-react-parser');

    //console.log(texts);
  
    const replaceInText = (e) => {//Complete ure done with displaying words in text
        let selectedText = window.getSelection().toString().trim();
        //var selRange = selectedText.getRangeAt(0);
        if(selectedText){
            console.log(selectedText);
            console.log(e);
        }
        //console.log()
    }
    const annotateText = () => {

        let newText = reactStringReplace(texts[id] , (match, i)=>(<Word word={match}/>))
        //let newTemp = reactStringReplace(newText,` ${temp} ` , (match, i)=>(<Word word={match}/>))
        //console.log(datasets[0]);
        for (let [key, value] of Object.entries(datasets[0])) {
            // console.log(value)
            let CatColor = colors[key]
            value.forEach(entry => {
                newText = reactStringReplace(newText,` ${entry} ` , (match, i)=>(<Word category={key} color= {CatColor} word={match}/>))

            })
            
        }
   
       
        return newText;
    }
    const returnComp = () => {
        
        if(!collapsed)
            return <>
                    <div className='docView-container'>
                        <div className="uncollapsed">
                            <img className='uncollapsedIcon' src={arrowIcon} alt="collapse icon" onClick={()=>(setCollapsed(!collapsed))}/>
                            <h2>{`Text ${id+1}`}</h2>
                        </div>
                        <Menu/>
                        <div className='text-cont'  >
                            {
                                annotateText()
                            }
               
                            
                        </div>
            
                    </div>
                </>
        else
            return <>
                    <div className='collapsed'>
                        <img className='collapseIcon' src={arrowIcon} alt="collapse icon" onClick={()=>(setCollapsed(!collapsed))}/>
                        <h2>{`Text ${id+1}`}</h2>
    
                    </div>
                </>     
        
    }
    return (
        returnComp()
    )
}

export default DocView
