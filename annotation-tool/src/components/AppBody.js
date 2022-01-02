
import './AppBody.css';
import React, { useState ,useEffect} from 'react';
import TRow from './TRow';
import { AiFillDelete } from 'react-icons/ai';
import {MdOutlineAddBox} from 'react-icons/md'

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
    var count =1;



    const getHeader =(datasets) => {

        let headers = [];
        let id = 0;
        var map = datasets.map((td) => Object.entries(td));
            //console.log(Array.isArray(map));
        //}
        //console.log(datasets.length);
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
            var map = datasets.map((td) => Object.entries(td));
            
            let array = map[0];

            var labelValues = array.map(row=> (row[1]));
            var maxArray= labelValues[0].length;
            console.log(maxArray);
            var k=0
            for (let i = 0; i < labelValues.length; i++) {
                if (labelValues[i].length > maxArray) {
                    maxArray = labelValues[i].length;
                    k=i;
                }
                
            }
            data = labelValues[k].map((_, colIndex) => labelValues.map(row => row[colIndex]));
     
            //console.log(data)
            setTableData(data);
    }


    const  populateData =  (e)=> {
        getData(e)
            .then(setData(filesData))
            .then(setDocSelected(true) );
            
    }

    const delCategory = (id) => {
        //TODO what ure righting is not working 
        //assumption: edit the datasets variables then everything will be rerendered 
        //
        console.log(tableData);
        var k = [...tableData];
        k.pop(id);
        console.log(k);
    }
    useEffect(() => {

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
                <div className='tableContainer'>
                    <table >
                        <thead>
                            <tr>
                                <th>--</th>
                                {
                                    headers.map((head)=><th key={head.id }>{head.hName}<AiFillDelete id={head.id} className='delIcon' onClick={() => delCategory(head.id)}/> </th>)
                                }
                            </tr>
                        </thead>
                        <tbody>
                            <tr className='addContainer'>
                                <td>--</td>
                                {
                                    headers.map((_) => {
                                        return(
                                            <td>
                                                <input type="text"   />
                                                <MdOutlineAddBox className='addIcon'/>
                                            </td>
                                        )
                                    })
                                }
                            </tr>
                            {
                                
                                tableData.map((row)=>(<TRow count = {count++} row ={row} />))
                            }
                    
                        </tbody>
                    </table>
                </div>
            </div>
        )
}


export default AppBody
