import React, { useState, useEffect } from 'react';
import "./Employees.css";
import { useNavigate } from "react-router-dom";
import Navigation from '../Navigation/Navigation';
import authService from "../../services/auth.service";
import employeeService from "../../services/employee.service";

import PropTypes from 'prop-types';
import Box from '@mui/material/Box';
import Collapse from '@mui/material/Collapse';
import IconButton from '@mui/material/IconButton';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import { cyan } from '@mui/material/colors';
import { styled } from '@mui/system';
// import { forceUpdate } from 'react-redux';


    const StyledTableContainer = styled(TableContainer)({
        borderRadius: '10px',
        
    });


    const StyledTableRow = styled(TableRow)({
        // borderRadius: '100px',
        // marginBottom: '10%',
        backgroundColor: '#032539',
        borderBottom: `13px solid transparent`,
        });
    
        const StyledTableCell = styled(TableCell)({
            color: 'white',
            border: 'none',
        });

    const StyleTableHeader = styled(TableRow)({
        boxShadow: '0px 3px 5px -1px rgba(0,0,0,0.2), 0px 6px 10px 0px rgba(0,0,0,0.14), 0px 1px 18px 0px rgba(0,0,0,0.12)',
        border: 'none',
        color: 'white'
        
    })


    function Row(props){
        const {row} = props;
        const [open, setOpen] = React.useState(false);
        const [tasks, setTasks] = useState([]); 

        useEffect(() => {
            const fetchData = async () => {
                const response = await employeeService.viewAllTasksByEmployee(row.id);
                const data = response.data;
                if (Array.isArray(data)) {
                setTasks(data);
                } else {
                console.error('Response data is not an array:', data);
                }
            };
            console.log('try1');
            fetchData();
            console.log('try2');
            }, [row.id]);
    

        return (
            <React.Fragment>
                <StyledTableRow spacing={16} sx={{ '& > *': { marginBottom: 50 } }} >
                    <StyledTableCell sx={{
                        borderTopLeftRadius: 30,
                        borderBottomLeftRadius: 30,
                        }}>
                        <IconButton 
                            aria-label="expand row"
                            size="small"
                            onClick={() => setOpen(!open)}
                        >
                            {open ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
                        </IconButton>
                    </StyledTableCell>
                    <StyledTableCell component="th" scope="row">
                        {row.firstname} {row.lastname}
                    </StyledTableCell>
                    <StyledTableCell align="right">{row.gender}</StyledTableCell>
                    <StyledTableCell align="right">{row.email}</StyledTableCell>
                    <StyledTableCell align="right">{row.businessUnit}</StyledTableCell>
                    <StyledTableCell align="right">{row.joinedDate}</StyledTableCell>
                    <StyledTableCell align="right">{row.terminatedDate}</StyledTableCell>
                    <StyledTableCell align="right">{row.location}</StyledTableCell>
                    <StyledTableCell align="right">{row.riskRating}</StyledTableCell>
                    <StyledTableCell sx={{ borderTopRightRadius: 30, borderBottomRightRadius: 30}} align="right">
                        {row.suspectedCases}
                    </StyledTableCell>
                </StyledTableRow>
                <TableRow>
                    <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
                        <Collapse in={open} timeout="auto" unmountOnExit>
                            <Box sx={{ margin: 1 }}>
                                <Typography variant="h6" gutterBottom component="div">
                                    <b>Incident History</b>
                                    
                                </Typography>
                                <Table size="small" aria-label="purchases">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell> <b>Incident Title</b> </TableCell>
                                            <TableCell> <b>Incident Timestamp</b></TableCell>
                                            <TableCell> <b>Severity</b> </TableCell>
                                            <TableCell align="right"> <b>Status</b> </TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {tasks.map((task) => (
                                            <TableRow key={task.id}>
                                            <TableCell>{task.incidentTitle}</TableCell>
                                            <TableCell>{task.incidentTimestamp}</TableCell>
                                            <TableCell>{task.severity}</TableCell>
                                            <TableCell align="right">{task.status}</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </Box>
                        </Collapse>
                    </TableCell>
                </TableRow>
            </React.Fragment>
        );


    }

    export default function Employees() {
        const color = cyan[500];
        const navigate = useNavigate();
        // State for all employees
        const [allEmployees, setAllEmployees] = useState([]);
        // State for filtered employees
        const [filteredEmployees, setFilteredEmployees] = useState([]);
        const [searchInput, setSearchInput] = useState(''); 
        const [selectedValue, setSelectedValue] = useState('');

        // Step 2
        const handleSearchInputChange = (event) => {
            setSearchInput(event.target.value);
            console.log("tracks typing");
        };

        // Step 3
        const handleSearchSubmit = async (event) => {
            event.preventDefault();
            // Fetch all employees
            const { data } = await employeeService.viewAllEmployees();
            setAllEmployees(data);
            // Filter employees based on the search input
            const filtered = data.filter((employee) =>
                employee.firstname.toLowerCase().includes(searchInput.toLowerCase())
            );
            // Update the state with the filtered employees
            setFilteredEmployees(filtered);
            
        };

        const handleSorting = async(event) => {
            setSelectedValue(event.target.value);
            console.log(event.target.value);
            if (event.target.value === "option1") {
                const { data } = await employeeService.viewAllEmployeesByRiskRating();
                setAllEmployees(data);
                setFilteredEmployees(data);
            }
            else {
                const { data } = await employeeService.viewAllEmployeesByCaseNumber();
                setAllEmployees(data);
                setFilteredEmployees(data);
            }
        };



        useEffect(() => {
            console.log("this is the state of employees now");
            console.log(filteredEmployees);
        }, [filteredEmployees]);




        const fetchEmployees = async (e) => {
            const { data } = await employeeService.viewAllEmployees();
            const employees = data;
            setAllEmployees(data);
            setFilteredEmployees(data);
            console.log("hello1");
            console.log(employees);
            console.log("hello2");

        };

        useEffect(() => {
            fetchEmployees();
            try{
                employeeService.viewAllEmployees().then(
                    () => {
                        console.log("ok");
                    },
                    (error) => {
                        console.log("Private page", error.response);
                        if (error.response && error.response.status === 403) {
                            authService.logout();
                            navigate("/auth/login");
                            window.location.reload();
                        }
                    }
                );

            }catch(err){
                console.log(err);
                authService.logout();
                navigate("/auth/login");
                window.location.reload();
            }
        }, []);

        

        return (
            <div className="Employee">
                <div className= "div">
                    <Navigation resetEmployees={fetchEmployees}/>
                </div>
                <div className = "headingEmployee">
                    <div className="title">
                        <div className="title-text">Employee Information</div>
                    </div>
                    <select className="dropDown" value={selectedValue} onChange={handleSorting}>
                        <option value="">Sort By</option>
                        <option value="option1">Risk Rating</option>
                        <option value="option2">Number Of Cases</option>
                    </select>

                    <div>
                        <form onSubmit={handleSearchSubmit}>
                            <input
                            type="text"
                            className="searchbar"
                            placeholder=" Search"
                            value={searchInput}
                            onChange={handleSearchInputChange}
                            />
                            <button type="submit" className="searchButton">Search</button>
                        </form>
                        {/* <input type="text" className="searchbar" placeholder=" Search" /> */}
                    </div>
                </div>
                <StyledTableContainer className="TableContainer" component={Paper}>
                    <Table sx={{ border: 'none'}} aria-label="collapsible table">
                        <TableHead>
                        <StyleTableHeader  >
                            <TableCell />
                            <TableCell>Name</TableCell>
                            <TableCell align="right"> <b>Gender</b>  </TableCell>
                            <TableCell align="right"> <b>Email</b>  </TableCell>
                            <TableCell align="right"> <b>Business Unit</b>  </TableCell>
                            <TableCell align="right"> <b>Joined Date</b> </TableCell>
                            <TableCell align="right"> <b>Terminated Date</b> </TableCell>
                            <TableCell align="right"> <b>Location</b> </TableCell>
                            <TableCell align="right"> <b>Risk Rating</b> </TableCell>
                            <TableCell align="right"> <b>Cases</b> </TableCell>
                        </StyleTableHeader>
                        </TableHead>
                        <TableBody sx={{ border: 'none'}} >
                            {filteredEmployees && filteredEmployees.length > 0 ? (
                                filteredEmployees.map((employee) => <Row key={employee.id} row={employee} />)
                            ) : (
                                <TableRow>
                                    <TableCell colSpan={11}>
                                        {searchInput.trim() === ''
                                            ? 'No employees found.'
                                            : 'No matching employees found.'}
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </StyledTableContainer>

            </div>
            
        );
    }