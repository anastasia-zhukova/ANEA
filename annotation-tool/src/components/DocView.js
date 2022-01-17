import React from 'react'
import { useState } from 'react/cjs/react.development';
import './DocView.css';
import arrowIcon from '../assets/arrow-white.svg';
import reactStringReplace from 'react-string-replace'
import Word from './Word';

import Menu from './Menu';





const DocView = ({setCrntCat, setCrntTerm, text, id, texts, setTexts, datasets, setDatasets, colors, getSelection, setDel, del, setChange}) => {
    const [collapsed, setCollapsed] = useState(true);
  
    //var parse = require('html-react-parser');

    //console.log(texts);
  
    
    const annotateText = () => {

        let newText = reactStringReplace(text , (match, i)=>(<Word word={match}/>))
        //let newTemp = reactStringReplace(newText,` ${temp} ` , (match, i)=>(<Word word={match}/>))
        //console.log(datasets[0]);
        let catId = 0
        for (let [key, value] of Object.entries(datasets[0])) {
            let CatColor = colors[key]
            value.forEach(entry => {
                newText = reactStringReplace(newText,` ${entry} ` , (match, i)=>(<Word 
                                                                                    setCrntCat= {setCrntCat}
                                                                                    setCrntTerm={setCrntTerm} 
                                                                                    setDatasets={setDatasets} datasets={datasets} 
                                                                                    setDel={setDel} del={del} 
                                                                                    setChange= {setChange} 
                                                                                    catId={catId} category={key} 
                                                                                    color= {CatColor} word={match}
                                                                                    />))
                    //duplicate but after you optimise your shit
 

            })
            catId++;
            
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
                        {/* <Menu datasets={datasets}/> */}
                        <div className='text-cont' onMouseUp={(e)=>(getSelection(e))} >
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
