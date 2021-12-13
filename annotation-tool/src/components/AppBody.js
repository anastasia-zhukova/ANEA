
import './AppBody.css'
import React, { useState } from 'react'
import { useEffect } from 'react';
//import { useState } from 'react';






var filesData = [];

const getData = async(e) => {
    let files = e.target.files;
    filesData.splice(0, filesData.length)

    for (let i = 0; i < files.length; i++) {

        let fr = new FileReader();
        fr.readAsText(files[i]);
        fr.onload =   () => {
            let fileContent = fr.result;
            //filesData= [...filesData, JSON.parse(fileContent)] ;
            filesData.push(JSON.parse(fileContent));
        }        
}
}


//TODO ()=> returnnew Promise((resolve, reject) => {the function})



const AppBody = ({datasets, setData}) => {
    const [docSelected, setDocSelected] = useState(false);

    const  populateData =  (e)=> {
        getData(e)
            .then(setData(filesData))
            .then(setDocSelected(true));
            
    }
    useEffect(() => {
        console.log(datasets);

    }, [datasets]);


    if (!docSelected) 
        return (
            <div className='body-container'>
                <h1> To start please choose a document or a dataset</h1>
                <div className='inputs-container'>
                    <input type='file' multiple={false} id='doc-input' accept='.pdf, .doc, .docx'/>
                    <label htmlFor='doc-input'>
                        Select document 
                    </label>

                    <input type='file' multiple={true} id='data-input' accept='.json' onChange= {populateData} />
                    <label htmlFor='data-input'>
                        Select dataset
                    </label>
                    {/* <button onClick={()=> console.log(datasets)} >click me </button> */}
                    
                </div>
        

            </div>
        )
    else
        return(
            <div className='body-container'>

            <h1>Dataset</h1>
            </div>
        )
}

export default AppBody
