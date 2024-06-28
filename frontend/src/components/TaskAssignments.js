import '../index.css';
import { useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { auth } from '../firebase-config';
import Sidebar from '../layouts/sidebar';
import tooltip from 'react-tooltip';
import Question from '../icon/question.svg';


const TaskAssignments = () => {
    const navigate = useNavigate();
    const [showNotification, setShowNotification] = useState(false);
    const [showPopup, setShowPopup] = useState(false);

    const [employeeFile, setEmployeeFile] = useState(null);
    const [taskFile, setTaskFile] = useState(null);
    const [licensedFile, setLicensedFile] = useState(null);
    const [overqualification, setOverqualification] = useState(false);
    const [maxEmployeeWorkload, setMaxEmployeeWorkload] = useState(null);
    const [weightIdle, setWeightIdle] = useState(null);
    const [weightScore, setWeightScore] = useState(null);
    const [weightBalance, setWeightBalance] = useState(null);
    const [presolve, setPresolve] = useState(null);
    const [heuristics, setHeuristics] = useState(null);
    const [MIPFocus, setMIPFocus] = useState(null);
    const [MIPGap, setMIPGap] = useState(null);
    const [threads, setThreads] = useState(null);

    const resetInputs = () => {
        setEmployeeFile(null);
        setTaskFile(null);
        setLicensedFile(null);
        setOverqualification(false);
        setMaxEmployeeWorkload(null);
        setWeightIdle(null);
        setWeightScore(null);
        setWeightBalance(null);
        setPresolve(null);
        setHeuristics(null);
        setMIPFocus(null);
        setMIPGap(null);
        setThreads(null);

        document.querySelectorAll('input[type="file"]').forEach(input => input.value = '');
        document.querySelectorAll('input[type="number"]').forEach(input => input.value = '');
        document.querySelectorAll('input[type="radio"]').forEach(input => input.checked = false);
    };
    
    const handleFileUpload = async () => {
        if (!employeeFile || !taskFile || !licensedFile) {
            console.error('employee, task, and license files must be selected');
            alert('employee, task, and license files must be selected');
            return;  
        }

        console.log("Uploading files...")

        setTimeout(() => {
            setShowNotification(true);
        }, 2000);

        const formData = new FormData();
        formData.append('employeeFile', employeeFile);
        formData.append('taskFile', taskFile);
        formData.append('licenseFile', licensedFile);
        formData.append('overqualification', document.querySelector('input[name="overqualification"]:checked').value);
        formData.append('maxEmployeeWorkload', document.querySelector('input[placeholder="20"]').value);
        formData.append('weightIdle', document.querySelector('input[placeholder="0"]').value);
        formData.append('weightScore', document.querySelector('input[placeholder="0"]').value);
        formData.append('weightBalance', document.querySelector('input[placeholder="0"]').value);
        formData.append('presolve', document.querySelector('input[placeholder="2"]').value);
        formData.append('heuristics', document.querySelector('input[placeholder="0.8"]').value);
        formData.append('MIPFocus', document.querySelector('input[placeholder="1"]').value);
        formData.append('MIPGap', document.querySelector('input[placeholder="0.01"]').value);
        formData.append('threads', document.querySelector('input[placeholder="2"]').value);


        try {
            const idToken = await auth.currentUser.getIdToken(true);
            const response = await fetch('https://dev-task-backend-d7nbrinueq-et.a.run.app/upload', {
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
            alert('Files uploaded successfully');
            // resetInputs();
        } catch (error) {
            console.error('Error:', error.message);
            alert('Upload failed: ' + error.message);
            // resetInputs();
        }

    };

    useEffect(() => {
        if (showNotification) {
            const timer = setTimeout(() => {
                setShowNotification(true);
            }, 3000);
            return () =>
                clearTimeout(timer);
        }
    }, [showNotification]);

    const Notification = () => {
        <div style={{ position: 'fixed', top: '10%', right: '10%', backgroundColor: 'lightgreen', padding: '10px', borderRadius: '5px', zIndex: 1000 }}>
            Files uploaded successfully!
        </div>
    }

    const [showTooltip, setShowTooltip] = useState(false);

    return (
        <div className="bg-gray-100 min-h-screen flex">
            <Sidebar />
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
                            {/* <div className='flex items-center justify-between'> */}
                                <label className='flex-1'>
                                    Max Employee Workload <span 
                                    onMouseEnter={() => setShowTooltip(true)}
                                    onMouseLeave={() => setShowTooltip(false)}>
                                        <button className='ml-2'>?</button>
                                        {showTooltip && (
                                            <div className='absolute bg-gray-700 text-white text-xs rounded p-2'>
                                            Informasi mengenai beban kerja maksimal karyawan.
                                            </div>
                                        )}
                                        </span>
                                </label>
                            {/* </div> */}
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
                            <div>

                                {/* <textarea
                                    className="w-full h-32 p-3 mt-1 border rounded"
                                    placeholder="Paste your license data here"
                                    onChange={e => setLicensedFile(e.target.value)}
                                >
                                </textarea> */}
                            </div>  
                        </div>
                    </div>
                        <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mt-4" onClick={handleFileUpload}>
                        Generate
                        </button>
                    </div>
                </div>
                </div>
            </div>
        </div>
    );
}

export default TaskAssignments;
