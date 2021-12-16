
import './AppBody.css'
import React, { useState ,useEffect} from 'react'
import TRow from './TRow';
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
        
            filesData.push(JSON.parse(fileContent));
        }        
    }
}






const AppBody = ({datasets, setData}) => {
    const [docSelected, setDocSelected] = useState(false);
    const [headers, setHeaders] = useState([]);
    const [tableData, setTableData] = useState([]);

    const getHeader =(datasets) => {

        let headers = [];
        let id = 0;
        var map = datasets.map((td) => Object.entries(td));
            //console.log(Array.isArray(map));
        //}
        console.log(datasets.length);
        if(map.length)
            map[0].forEach(row => {
                
                headers = [...headers, {id:id++, hName: row[0]}];
            });
        // for (let i = 0; i < map[0].length; i++) {
        //     headers = [...headers, map[0][i]];
            
            
        // }    
        //console.log(map);
        setHeaders(headers);
    
    }
    const getRows = (datasets) => {
        let data = [];
        let row = [];
            var map = datasets.map((td) => Object.entries(td));
            console.log(map);
            let array = map[0];
           // console.log(array);
           //TODO we need to check which column is longer 
           //TODO we don't get all the data cz we use vertical length 
           //TODO improuuuve this function 
            for (let i = 0; i < array.length; i++) {
                for (let j = 0; j < array[i][1].length; j++) {
                    //console.log(array[j]?.[1]?.[i])
                    row.push(array[j]?.[1]?.[i])
                }
                //console.log("\n");
                data.push(row);
                row=[]
            }
    
    
            console.log(data)
            setTableData(data);
    }
    const  populateData =  (e)=> {
        getData(e)
            .then(setData(filesData))
            .then(setDocSelected(true) );
            
    }
    useEffect(() => {
        //console.log(datasets[0]);

        //console.log(datasets);
        getHeader(datasets);
  
    },[datasets]);


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
                <button onClick={() => {getHeader(datasets); getRows(datasets); }} className='temp-btn'>Show dataset</button>

                <table>
                    <thead>
                        <tr>
                            {
                                headers.map((head)=><th key={head.id }>{head.hName}<button>Del</button></th>)
                            }
                        </tr>
                    </thead>
                    <tbody>
                        {
                            tableData.map((row)=>(<TRow row ={row} />))
                        }
                 
                    </tbody>
                </table>
            </div>
        )
}

export default AppBody
