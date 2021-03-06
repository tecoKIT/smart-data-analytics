/* Apply K-Means clustering to customer dataset */

/* Define table types that will be used in the script */
CREATE TYPE PAL_T_KM_DATA AS TABLE (ID INTEGER, INCOME DOUBLE, LOYALTY DOUBLE);
CREATE TYPE PAL_T_KM_PARAMS AS TABLE (NAME VARCHAR(60), INTARGS INTEGER, DOUBLEARGS DOUBLE, STRINGARGS VARCHAR (100));
CREATE TYPE PAL_T_KM_RESULTS AS TABLE (ID INTEGER, CENTER_ID INTEGER, DISTANCE DOUBLE);
CREATE TYPE PAL_T_KM_CENTERS AS TABLE (CENTER_ID INTEGER, INCOME DOUBLE, LOYALTY DOUBLE);

/* Table to generate K-Means procedure */
CREATE COLUMN TABLE PAL_KM_SIGNATURE (ID INTEGER, TYPENAME VARCHAR(1000), DIRECTION VARCHAR(100));
INSERT INTO PAL_KM_SIGNATURE VALUES (1, 'UTNSU.PAL_T_KM_DATA', 'in');
INSERT INTO PAL_KM_SIGNATURE VALUES (2, 'UTNSU.PAL_T_KM_PARAMS', 'in');
INSERT INTO PAL_KM_SIGNATURE VALUES (3, 'UTNSU.PAL_T_KM_RESULTS', 'out');
INSERT INTO PAL_KM_SIGNATURE VALUES (4, 'UTNSU.PAL_T_KM_CENTERS', 'out');

/* Create K-Means Procedure with ALF wrapper generator */
CALL SYSTEM.AFL_WRAPPER_GENERATOR ('PAL_KM', 'AFLPAL', 'KMEANS', PAL_KM_SIGNATURE);

/* Create view with only income and loyalty */
CREATE VIEW V_KM_DATA AS 
	SELECT ID, INCOME, LOYALTY
		FROM CUSTOMERS;
		
/* Create and fill parameter table */
CREATE COLUMN TABLE KM_PARAMS LIKE PAL_T_KM_PARAMS;
INSERT INTO KM_PARAMS VALUES ('THREAD_NUMBER', 2, null, null);
INSERT INTO KM_PARAMS VALUES ('GROUP_NUMBER', 3, null, null);
INSERT INTO KM_PARAMS VALUES ('INIT_TYPE', 1, null, null);
INSERT INTO KM_PARAMS VALUES ('DISTANCE_LEVEL', 2, null, null);
INSERT INTO KM_PARAMS VALUES ('MAX_ITERATION', 100, null, null);
INSERT INTO KM_PARAMS VALUES ('NORMALIZATION', 0, null, null);
INSERT INTO KM_PARAMS VALUES ('EXIT_THRESHOLD', null, 0.0001, null);

/* Create result tables */
CREATE COLUMN TABLE KM_RESULTS_FROM_SCRIPT LIKE PAL_T_KM_RESULTS;
CREATE COLUMN TABLE KM_CENTER_POINTS_SCRIPT LIKE PAL_T_KM_CENTERS;

/* create some views to analyze later the results */
CREATE VIEW V_KM_RESULTS_FROM_SCRIPT AS
	SELECT a.ID, b.CUSTOMER,  b.INCOME, b.LOYALTY, a.CENTER_ID + 1 AS CLUSTER_NUMBER
		FROM KM_RESULTS_FROM_SCRIPT a, CUSTOMERS b 
		WHERE a.ID = b.ID;
		
/* call K-Means procedure */
CALL _SYS_AFL.PAL_KM (V_KM_DATA, KM_PARAMS, KM_RESULTS_FROM_SCRIPT, KM_CENTER_POINTS_SCRIPT) WITH OVERVIEW;


------------------------------------------------
-- K-Mean Validation
------------------------------------------------

/* Define table types */
CREATE TYPE PAL_T_KM_V_TYPE_ASSIGN AS TABLE (ID INTEGER, TYPE_ASSIGN INTEGER);
CREATE TYPE PAL_T_KM_V_RESULTS AS TABLE (NAME VARCHAR(50), S DOUBLE);

CREATE COLUMN TABLE PAL_KM_V_SIGNATURE (ID INTEGER, TYPENAME VARCHAR(100), DIRECTION VARCHAR(100));
INSERT INTO PAL_KM_V_SIGNATURE VALUES (1, 'UTNSU.PAL_T_KM_DATA', 'in');
INSERT INTO PAL_KM_V_SIGNATURE VALUES (2, 'UTNSU.PAL_T_KM_V_TYPE_ASSIGN', 'in');
INSERT INTO PAL_KM_V_SIGNATURE VALUES (3, 'UTNSU.PAL_T_KM_PARAMS', 'in');
INSERT INTO PAL_KM_V_SIGNATURE VALUES (4, 'UTNSU.PAL_T_KM_V_RESULTS', 'out');
	
/* Create K-Means validation Procedure with ALF wrapper generator */	
CALL SYSTEM.AFL_WRAPPER_GENERATOR ('PAL_KM_V', 'AFLPAL', 'VALIDATEKMEANS', PAL_KM_V_SIGNATURE);

CREATE VIEW V_KM_TYPE_ASSIGN AS 
	SELECT ID, CENTER_ID AS TYPE_ASSIGN 
		FROM KM_RESULTS_FROM_SCRIPT;

/* Fill the Parameters Table */
CREATE COLUMN TABLE KM_V_PARAMS LIKE PAL_T_KM_PARAMS;
CREATE COLUMN TABLE KM_V_RESULTS LIKE PAL_T_KM_V_RESULTS;
INSERT INTO KM_V_PARAMS VALUES ('VARIABLE_NUM', 2, null, null);
INSERT INTO KM_V_PARAMS VALUES ('THREAD_NUMBER', 1, null, null);

/* Call the Validate KMeans procedure */
CALL _SYS_AFL.PAL_KM_V (V_KM_DATA, V_KM_TYPE_ASSIGN, KM_V_PARAMS, KM_V_RESULTS) WITH OVERVIEW;


	
	
