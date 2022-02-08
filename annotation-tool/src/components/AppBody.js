
import './AppBody.css';
import React, { useRef, useState } from 'react';
import GridView from './GridView';
import Category from './Category';
import {MdOutlineAddBox} from 'react-icons/md'

import DocView from './DocView';
import Menu from './Menu';
//import { useEffect } from 'react/cjs/react.development';

// import TRow from './TRow';
// import { AiFillDelete } from 'react-icons/ai';
// import {MdOutlineAddBox} from 'react-icons/md'
// import Tdata from './Tdata';
// //import  { useRef } from 'react'

//import { useState } from 'react';
var filesData = [];
var textsInFile = "";
var fileName = "";


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

const getData = (e) => {
    return new Promise((resolve, reject)=>{

        let files = e.target.files;
        fileName = files[0].name
        filesData.splice(0, filesData.length)
        try{
            for (let i = 0; i < files.length; i++) {
                let fr = new FileReader();
                fr.readAsText(files[i]);
                fr.onload = () => {
                    filesData.push(JSON.parse(fr.result));
                    resolve('ok');
                }  
            }
        }catch(err){
            console.error(err);
                    reject(err);
        }

    })
}





const AppBody = () => {

    
    const [datasets, setDatasets] = useState([{}]);
    const [texts, setTexts] = useState([]);
    const [textSelected, setTextSelect] = useState(false);
    const [docSelected, setdocSelect] = useState(false);
    const [isGrid, setGrid] = useState(false);
    const [CatColors, SetColors] = useState({});
    const [del, setDel] = useState(false);
    const [change, setChange] = useState(false);
    const [add, setAdd] = useState(false);
    const [selectedTxt, setText] = useState("");
    const [currentTerm, setCrntTerm] =useState("");
    const [currentCat, setCrntCat] = useState();
    let addCatInput = useRef();

    const delColor= (name) => {
        let newColors = CatColors;
        delete newColors[name];
        SetColors({...newColors})

    }
    const checkEnter = (e) => {
        if(e.keyCode !== 13) return;
        addCategory();
    }
    const  addCategory = () => {
        let newData = datasets;
        
        let catName = addCatInput.current.value.trim();
        if (catName.length === 0) return;

        newData[0][catName] = [];
        addCatColor(catName);
        setDatasets([...newData]);
        addCatInput.current.value ="";



    }
    const  populateData = async (e)=> {

            await getData(e);
            await setDatasets(filesData);
            setdocSelect(!docSelected);
            generatColors();        
    }
    
    const onchangeHandel = async  (e)=>{
        try{
            await getText(e);            
            e.target.value = null;
            setTexts(textsInFile);
            setTextSelect(!textSelected)
        }catch(err){
            alert("something went wrong please check the file format!!")
            //console.log("hh")
        }
        
    };

    function getRandomColor() {
        var letters = '0123456789ABCDEF';
        var color = '#';
        for (var i = 0; i < 6; i++) {
          color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
      }
    const addCatColor = (catName) => {
        let newColors = CatColors;
        newColors[catName] = getRandomColor();
        SetColors({...newColors});
    }
    const generatColors = ()=>{
        
        Object.keys(filesData[0]).map((cat)=>{
            let newColors = CatColors;
            newColors[cat] = getRandomColor();

            SetColors({...newColors});// u may need an await statement 
        })
    }

    const getSelection = (e) => {
        let selectedText = window.getSelection().toString().trim();
        //var selRange = selectedText.getRangeAt(0);
        if(selectedText){
            setCrntTerm(selectedText);
            setText(selectedText);
            setAdd(true);
            return;
        }
        setAdd(false);
        

        
        //console.log()
    }
    const returnCatsLegend =() => {
        var keys = Object.keys(CatColors);
        return(
            //console.log(CatColors);
            keys.map((key, index)=>(
               // console.log(CatColors[key]);
                <Category 
                    key={index}
                    delColor ={delColor}
                    color={CatColors[key]} 
                    catName={key}  catId ={index}
                    datasets = {datasets} setDatasets= {setDatasets}
                    />    
            ))
        );
    }
    const downloadObjectAsJson = (exportObj, exportName) => {
        var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportObj));
        var downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", exportName);
        document.body.appendChild(downloadAnchorNode); // required for firefox
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    }
    const exportData = () => {
        if (fileName)
            downloadObjectAsJson(datasets[0], `annotated ${fileName}`);
        else
            downloadObjectAsJson(datasets[0], `annotated dataset.json`);

      //console.log(datasets[0]);
    }


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
                        <GridView  delColor={delColor} addCatColor={addCatColor} datasets={datasets} setDatasets={setDatasets}/>
                        <div className="switch-cont">
                            <button className='export-btn' onClick={exportData} > Export dataset </button>
                            <button onClick={()=>(setGrid(!isGrid))} > Document view </button>
                        </div>
                    </>
                else  {  
                  
                return <>
                    <div className="docView-cont"  >
                        <Menu  
                            selectedTxt={selectedTxt} 
                            datasets={datasets} 
                            setDatasets={setDatasets} 
                            del={del} change={change} add={add} 
                            setAdd={setAdd} setChange={setChange} setDel={setDel}
                            currentTerm= {currentTerm} 
                            currentCat = {currentCat}   
                        />

                        <div className='cat-cont'>
                            <div>
                                <h3>Categories</h3>
                                <div className='valueContainer'>
                                    <div className='addCatCont'>
                                        <input type="text" ref={addCatInput} onKeyDown={(e)=>checkEnter(e)}/>
                                        <MdOutlineAddBox color="#00ab95" className='addIcon' onClick={addCategory}/>
                                    </div>
                                </div>
                            </div>
                            {
                                returnCatsLegend()
                            }
                        </div>
                        
                        <div className='texts-cont'>
                            <h1> Your texts: </h1>
                            {
                                texts.map((text, index)=>(<DocView key={index} id={index} 
                                    setDel={setDel} del={del} 
                                    setChange= {setChange}
                                    getSelection={getSelection} colors = {CatColors} 
                                    datasets={datasets} setDatasets={setDatasets} 
                                    text = {text} setTexts = {setTexts} 
                                    setCrntTerm= {setCrntTerm}
                                    setCrntCat = {setCrntCat}
                                    />))
                            }
                            
                            <div className="switch-cont switch-contDoc">
                                <button className='export-btn' onClick={exportData} > Export dataset </button>
                                <button onClick={()=>(setGrid(!isGrid))} > Grid view</button>
                            </div>
                        </div>
                    </div>
                </>
                }
            else    // to give the user the choice to choose between an existing dataset or to annotate from scratch 
                return <> 
                    <h1>Do you have an existing Dataset or you want to annotate from scratch?</h1>
                    <div className='inputs-container'>
                        <input type='file' multiple={true} id="data-input" accept='.json'  onChange={populateData}/>
                        <label className='annotate-lbl'  htmlFor='data-input'>
                            Select dataset
                        </label>
                        <button onClick={()=>(setdocSelect(!docSelected))} className='annotate-btn'>Annotate from scratch</button>
                    </div>
                </>  


        }
      
    }

    // useEffect(()=>{
    //     returnItem();
   
    // }, [datasets]);
 
    return (

        <div className='body-container'>
            
           {returnItem()} 


        </div>
    )
}

export default AppBody
