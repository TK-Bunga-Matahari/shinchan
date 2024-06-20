import '../index.css';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { auth } from '../firebase-config';

const TaskAssignments = () => {
    const navigate = useNavigate();
    const [showPopup, setShowPopup] = useState(false);
    const [userId, setUserId] = useState('');

    const handleMenu = () => {
        navigate("/menu");
    }

    const handleResult = () => {
        navigate("/menu/result");
    }

    const handleGenerate = async () => {
        // Simulate fetching user ID from backend
        const fetchedUserId = '12345';  // This should be replaced with actual API call
        setUserId(fetchedUserId);
        setShowPopup(true);
    }

    const closePopup = () => {
        setShowPopup(false);
    }

    const handleProgress = () => {
        navigate("/menu/progress");
    }

    const [employeeFile, setEmployeeFile] = useState(null);
    const [taskFile, setTaskFile] = useState(null);
    const [licensedFile, setLicensedFile] = useState(null);

    const handleFileUpload = async () => {
        if (!employeeFile || !taskFile || !licensedFile) {
            console.error('Both employee, task, license files must be selected');
            return;
        }

        const formData = new FormData();
        formData.append('employeeFile', employeeFile);
        formData.append('taskFile', taskFile);
        formData.append('licensedFile', licensedFile);

        try {
            const idToken = await auth.currentUser.getIdToken(true);
            const response = await fetch('https://my-backend-ev6a2mrxbq-et.a.run.app/upload', {
                method: 'POST',
                body: formData,
                headers: {
                    'Authorization': 'Bearer ' + idToken
                }
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            console.log('Files uploaded successfully:', data);
        } catch (error) {
            console.error('Error:', error.message);
        }
    };

    return (
        <div className="bg-gray-100 min-h-screen flex">
            {/* Sidebar */}
            <div className="w-60 bg-white shadow-md px-5 py-7">
                <h1 className="text-xl font-semibold mb-10">Optimization</h1>
                <ul className="space-y-4">
                    <li>
                        <button
                        className="text-gray-700 hover:text-gray-900 bg-transparent border-none cursor-pointer"
                        onClick={handleMenu}
                        >
                        Toolbox
                        </button>
                    </li>
                    <li>
                        <button
                        className="text-gray-700 hover:text-gray-900 bg-transparent border-none cursor-pointer"
                        onClick={handleResult}
                        >
                        Results
                        </button>
                    </li>
                    <li>
                        <button
                        className="text-gray-700 hover:text-gray-900 bg-transparent border-none cursor-pointer"
                        onClick={handleProgress}
                        >
                            Track progress
                        </button>
                    </li>
                </ul>
            </div>

            {/* Main Content */}
            <div className="flex-grow p-10">
                <div className="max-w-4xl mx-auto">
                <h2 className="text-2xl font-bold mb-6">Task Assignment</h2>
                <div className="flex gap-10">
                    {/* Parameter section */}
                    <div className="bg-white shadow-lg p-6 flex-1 rounded">
                    {/* <h3 className="text-lg font-semibold mb-4">Overview</h3> */}
                    <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                            <label className="flex-1">Overqualification:</label>
                            <label>
                                <input type="radio" name="overqualification" value="true" className="mr-2" /> True
                            </label>
                            <label>
                                <input type="radio" name="overqualification" value="false" className="mr-2" /> False
                            </label>
                        </div>
                        <div>
                            <label >Max Employee Workload</label>
                            <input type="number" className="border-2 border-black rounded w-full" step="0.01" placeholder="20"/>
                        </div>
                        <div>
                            <label>Weight Idle</label>
                            <input type="number" className="border-2 border-black rounded w-full" step="0.01" placeholder="0" />
                        </div>
                        <div>
                            <label>Weight Score</label>
                            <input type="number" className="border-2 border-black rounded w-full" step="0.01" placeholder="0" />
                        </div>
                        <div>
                            <label>Weight Balance</label>
                            <input type="number" className="border-2 border-black rounded w-full" step="0.01" placeholder="0" />
                        </div>
                        <div>
                            <label className='px-8'>Define the tuned parameters of the model</label>
                        </div>
                        <div>
                            <label>Presolve</label>
                            <input type="number" className="border-2 border-black rounded w-full" step="0.01" placeholder="2" />
                        </div>
                        <div>
                            <label>Heuristics</label>
                            <input type="number" className="border-2 border-black rounded w-full" step="0.01" placeholder="0.8" />
                        </div>
                        <div>
                            <label>MIPFocus</label>
                            <input type="number" className="border-2 border-black rounded w-full" step="0.01" placeholder="1" />
                        </div>
                        <div>
                            <label>MIPGap</label>
                            <input type="number" className="border-2 border-black rounded w-full" step="0.01" placeholder="0.01" />
                        </div>
                        <div>
                            <label>Threads</label>
                            <input type="number" className="border-2 border-black rounded w-full" step="0.01" placeholder="2" />
                        </div>
                    </div>
                    </div>

                    {/* File upload section */}
                    <div className="bg-white shadow-lg p-6 flex-1 rounded">
                    <h3 className="text-lg font-semibold mb-4">Upload Data</h3>
                    <div className="space-y-1">
                        {/* <div>
                            <label>Upload Employee Data</label>
                            <input type="file" className="mb-4 py-3" accept=".csv"/>
                        </div>
                        <div>
                            <label>Upload Task Data</label>
                            <input type="file" className="mb-4 py-3" accept=".csv"/>
                        </div>
                        <div>
                            <label>Upload Licensed File</label>
                            <input type="file" className="mb-4 py-3" accept=".lic"/>
                        </div> */}
                        <div>
                            <label>Upload Employee Data</label>
                            <input type="file" className="mb-4 py-3" accept=".csv" onChange={e => setEmployeeFile(e.target.files[0])} />
                        </div>
                        <div>
                            <label>Upload Task Data</label>
                            <input type="file" className="mb-4 py-3" accept=".csv" onChange={e => setTaskFile(e.target.files[0])} />
                        </div>
                        <div>
                            <label>Upload Licensed File</label>
                            <input type="file" className="mb-4 py-3" accept=".lic" onChange={e => setLicensedFile(e.target.files[0])} />
                        </div>
                    </div>
                    <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mt-4" onClick={handleFileUpload}>
                    Generate
                    </button>
                    </div>
                </div>
                </div>
            </div>
            {/* {showPopup && (
                <div className="absolute top-0 left-0 w-full h-full bg-black bg-opacity-50 flex justify-center items-center">
                    <div className="bg-white p-5 rounded flex flex-col items-center">
                        <p className="mb-4">Save your ID to track your progress.</p>
                        <h3 className="mb-4 text-center">{userId}</h3>
                        <button className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded" onClick={closePopup}>
                            Close
                        </button>
                    </div>
                </div>
            )} */}
        </div>
    );
}

export default TaskAssignments;
