// import React from 'react';
// import '../index.css';
// import { useState } from 'react';
// import image from '../output/similarity_all.png';
// import image2 from '../output/similarity_all copy.png';
// import { useNavigate } from 'react-router-dom';
// import Sidebar from '../layouts/sidebar';

// const Result = () => {
//     return (
//         <div className="bg-gray-100 min-h-screen flex">
//             <Sidebar />

//             {/* Main Content */}
//             <div className="flex-grow p-10">
//                 <div className="max-w-4xl mx-auto">
//                 <h2 className="text-2xl font-bold mb-6">Results</h2>
//                 <ResultsComponent />
//                 </div>
//             </div>
//         </div>
//     );
// };

// function ResultsComponent() {
//     const [showTable, setShowTable] = useState(false);
//     const [showImage, setShowImage] = useState(false);

//     return (
//         <div className="space-y-8">
//             {/* Card 1: Table */}
//             <div className="bg-white p-4 shadow-lg rounded-lg">
//             <h3 className="text-lg font-semibold mb-4">Employee Details</h3>
//             <div className="overflow-x-auto">
//                 {/* Placeholder for table */}
//                 <table className="min-w-full table-auto">
//                 <thead className="bg-gray-200">
//                     <tr>
//                     <th className="px-4 py-2">Employee</th>
//                     <th className="px-4 py-2">Assigned Task</th>
//                     <th className="px-4 py-2">Company</th>
//                     <th className="px-4 py-2">Assigned Story Point</th>
//                     <th className="px-4 py-2">Wasted Story Point</th>
//                     <th className="px-4 py-2">Similarity Score</th>
//                     </tr>
//                 </thead>
//                 <tbody>
//                     <tr>
//                     <td className="border px-4 py-2 text-center">Talent 101</td>
//                     <td className="border px-4 py-2 text-center">T31, T54, T65, T67, T89</td>
//                     <td className="border px-4 py-2 text-center">C3</td>
//                     <td className="border px-4 py-2 text-center">18</td>
//                     <td className="border px-4 py-2 text-center">2</td>
//                     <td className="border px-4 py-2 text-center">80%</td>
//                     </tr>
//                     {/* Additional rows can be added here */}
//                 </tbody>
//                 </table>
//             </div>
//             </div>
    
//             {/* Card 2: Image */}
//             <div className="bg-white p-4 shadow-lg rounded-lg">
//                 <h3 className="text-lg font-semibold mb-4">Plot Visualization</h3>
//                 <div className="flex flex-col justify-center items-center h-auto">
//                     {/* Placeholder for image */}
//                     <img src={image} alt="Placeholder" className="max-h-full" />
//                     {/* <img src={image2} alt="Placeholder" className="max-h-full" /> */}
//                     {/* <img src={image2} alt="Placeholder" className="max-h-full" /> */}
//                 </div>
//             </div>
//         </div>
//     );
// }

// export default Result;

//=========================================================//

import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import '../index.css'; // Ensure CSS path is correct
import Sidebar from '../layouts/sidebar'; // Ensure component path is correct

const Result = () => {
    const [data, setData] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch('../output/result.csv');
                if (!response.ok) throw new Error('Network response was not ok.');
                const text = await response.text(); // Read the entire response as text
                Papa.parse(text, {
                    header: true,
                    complete: function(results) {
                        if (results.errors.length) {
                            console.error('Errors parsing CSV:', results.errors);
                            return; // Stop processing if there are errors
                        }
                        // Process each row to convert stringified arrays back to JavaScript arrays
                        const processedData = results.data.map(row => ({
                            ...row,
                            company: JSON.parse(row.company.replace(/'/g, '"')), // Replace single quotes with double quotes and parse
                            assigned_task: JSON.parse(row.assigned_task.replace(/'/g, '"')),
                            similarity_score: JSON.parse(row.similarity_score.replace(/'/g, '"'))
                        }));
                        setData(processedData);
                    }
                });
            } catch (error) {
                console.error("Error fetching or parsing CSV data:", error);
            }
        };
    
        fetchData();
    }, []);

    return (
        <div className="bg-gray-100 min-h-screen flex">
            <Sidebar />

            {/* Main Content */}
            <div className="flex-grow p-10">
                <div className="max-w-4xl mx-auto">
                    <h2 className="text-2xl font-bold mb-6">Results</h2>
                    <div className="bg-white p-4 shadow-lg rounded-lg">
                        <h3 className="text-lg font-semibold mb-4">Employee Details</h3>
                        <div className="overflow-x-auto">
                            <table className="min-w-full table-auto">
                                <thead className="bg-gray-200">
                                    <tr>
                                        <th className="px-4 py-2">Employee</th>
                                        <th className="px-4 py-2">Company</th>
                                        <th className="px-4 py-2">Assigned Task</th>
                                        <th className="px-4 py-2">Assigned Story Point</th>
                                        <th className="px-4 py-2">Wasted Story Point</th>
                                        <th className="px-4 py-2">Similarity Score</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {data.map((item, index) => (
                                        <tr key={index}>
                                            <td className="border px-4 py-2 text-center">{item.employee}</td>
                                            <td className="border px-4 py-2 text-center">{item.assigned_task.join(', ')}</td>
                                            <td className="border px-4 py-2 text-center">{item.company.join(', ')}</td>
                                            <td className="border px-4 py-2 text-center">{item.sum_sp}</td>
                                            <td className="border px-4 py-2 text-center">{item.wasted_sp}</td>
                                            <td className="border px-4 py-2 text-center">{item.similarity_score.join(', ')}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Result;


// =========================================================//


// import React, { useState, useEffect } from 'react';
// import Papa from 'papaparse';
// import image from '../output/similarity_all.png';
// import image2 from '../output/similarity_all copy.png';
// import '../index.css';

// function Result() {
//     const [rows, setRows] = useState([]);

//     useEffect(() => {
//         // Fetching the CSV data
//         fetch('../output/result_obj1.csv')
//             .then(response => response.text())
//             .then(data => {
//                 Papa.parse(data, {
//                     header: true,
//                     complete: (results) => {
//                         setRows(results.data);
//                     }
//                 });
//             });
//     }, []);

//     return (
//         <div className="bg-gray-100 min-h-screen flex">
//             <div className="bg-white p-4 shadow-lg rounded-lg">
//                 <h3 className="text-lg font-semibold mb-4">Employee Details</h3>
//                 <div className="overflow-x-auto">
//                     <table className="min-w-full table-auto">
//                         <thead className="bg-gray-200">
//                             <tr>
//                                 <th className="px-4 py-2">Employee</th>
//                                 <th className="px-4 py-2">Company</th>
//                                 <th className="px-4 py-2">Assigned Task</th>
//                                 <th className="px-4 py-2">Assigned Story Point</th>
//                                 <th className="px-4 py-2">Wasted Story Point</th>
//                                 <th className="px-4 py-2">Similarity Score</th>
//                             </tr>
//                         </thead>
//                         <tbody className="max-h-64 overflow-y-auto">
//                             {rows.map((item, index) => (
//                                 <tr key={index}>
//                                     <td className="border px-4 py-2">{item.employee}</td>
//                                     <td className="border px-4 py-2">{Array.isArray(item.company) ? item.company.join(', ') : item.company}</td>
//                                     <td className="border px-4 py-2">{Array.isArray(item.assigned_task) ? item.assigned_task.join(', ') : item.assigned_task}</td>
//                                     <td className="border px-4 py-2">{item.sum_sp}</td>
//                                     <td className="border px-4 py-2">{item.wasted_sp}</td>
//                                     <td className="border px-4 py-2">
//                                         {Array.isArray(item.similarity_score) ? item.similarity_score.map(score => score.toFixed(2)).join(', ') : item.similarity_score.toFixed(2)}
//                                     </td>
//                                 </tr>
//                             ))}
//                         </tbody>
//                     </table>
//                 </div>
//             </div>
//             <div className="bg-white p-4 shadow-lg rounded-lg">
//                 <h3 className="text-lg font-semibold mb-4">Plot Visualization</h3>
//                 <div className="flex flex-col justify-center items-center h-auto">
//                     <img src={image} alt="Plot Visualization" className="max-h-full" />
//                     <img src={image2} alt="Additional Plot" className="max-h-full" />
//                 </div>
//             </div>
//         </div>
//     );
// }

// export default Result;
