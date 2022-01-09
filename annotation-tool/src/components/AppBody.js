
import './AppBody.css';
import React, { useState } from 'react';
import GridView from './GridView';
import { useEffect } from 'react/cjs/react.development';
import { resolvePath } from 'react-router-dom';
import DocView from './DocView';
// import TRow from './TRow';
// import { AiFillDelete } from 'react-icons/ai';
// import {MdOutlineAddBox} from 'react-icons/md'
// import Tdata from './Tdata';
// //import  { useRef } from 'react'

//import { useState } from 'react';
    





var filesData = [];
var textsInFile = "";

const getData = async(e) => {
    let files = e.target.files;
    filesData.splice(0, filesData.length)

    for (let i = 0; i < files.length; i++) {

        let fr = new FileReader();
        fr.readAsText(files[i]);
        fr.onload = () => {
            let fileContent = fr.result;
        
            filesData.push(JSON.parse(fileContent));
        }        
    }

}

const getText = (e) => {
    return new Promise((resolve, reject)=>{
        try{
            let file = e.target.files[0];
            let fr = new FileReader();
            fr.readAsText(file);
            fr.onload = () => {
                //textsInFile = ;

                textsInFile =  JSON.parse( fr.result);
                resolve("ok")
                //console.log(textsInFile);
            }

            
        }catch(err){
            //TODO handel with an exception 
            reject(err);
            

        }

    })



    
}




const AppBody = () => {

    const [datasets, setDatasets] = useState([]);
    const [texts, setTexts] = useState([]);
    const [textSelected, setTextSelect] = useState(false);
    const [docSelected, setdocSelect] = useState(false);
    const [isGrid, setGrid] = useState(false);

    const  populateData =  (e)=> {
        getData(e)
            .then(setDatasets(filesData))
            .then(setdocSelect(!docSelected));//setDocSelected(true) );
            
            
    }
    
    const onchangeHandel = async  (e)=>{
        try{
            await getText(e);            
            e.target.value = null;
            setTexts(textsInFile);
            setTextSelect(!textSelected)
        }catch(err){
            alert("something went wrong please check the file format!!")
        }
        
    };
//ToDO
// done with the grid view 
// next work on the text annotation : see the notes writen : put the text into spans and then iterate over the datasets 
// gd luck

    const returnItem = () => {
        if (!textSelected) ////when the text file is selected 
            return <>
                <h1> To start please choose your document</h1>
                <div className='inputs-container'>
                    <input type='file' multiple={false} id="doc-input" accept='.json' onChange={(e)=>{
                        onchangeHandel(e); 
                    }}/>
                    <label htmlFor='doc-input' className='doc-lbl'>
                        Select Text 
                    </label>
                </div>
            </>
        else{
            if(docSelected)// when the dataset file is selected 
                if(isGrid)
                    return <>
                        <GridView datasets={datasets} setDatasets={setDatasets}/>
                        <div className="switch-cont">
                            <button onClick={()=>(setGrid(!isGrid))} > Document view </button>
                        </div>
                    </>
                else    
                return <>
                    <div className="docView-cont"  >
                        <h1> Your texts: </h1>
                        {
                            texts.map((text, index)=>(<DocView key={index} id={index}  text={text} />))
                        }
                        
                        <div className="switch-cont">
                            <button onClick={()=>(setGrid(!isGrid))} > Grid view</button>
                        </div>
                    </div>
                </>
            else    // to give the user the choice to choose between an existing dataset or to annotate from scratch 
                return <> 
                    <h1>Do you have an existing Dataset or you want to annotate from scratch?</h1>
                    <div className='inputs-container'>
                        <input type='file' multiple={true} id="data-input" accept='.json'  onChange={populateData}/>
                        <label className='annotate-lbl'  htmlFor='data-input'>
                            Select dataset
                        </label>
                        <button className='annotate-btn'>Annotate from scratch</button>
                    </div>
                </>  


        }
      
    }
    // useEffect(()=>{
    //     console.log(texts);
    // },[texts]);
    //console.log(texts);
    return (

        <div className='body-container'>
            
           {returnItem()} 


        </div>
    )
}

export default AppBody
